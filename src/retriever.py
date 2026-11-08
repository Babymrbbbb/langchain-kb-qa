"""检索：chromadb 稠密向量召回 + BM25 词法重排（RRF 融合）。

为什么做混合检索（hybrid）：
纯向量召回（尤其是英文预训练 MiniLM，或降级用的哈希向量）在中文垂域的
语义/词法召回有限。BM25 在中文二元组分词下表现极好（姊妹项目
rag-vector-vs-keyword 实测 Recall@5=0.953）。这里用「向量召回候选 +
全库 BM25 召回」做 Reciprocal Rank Fusion（RRF），显著提升中文检索准确率，
且不引入任何额外依赖。检索层是 RAG 的硬骨头——召回准，后面的生成才有意义。
"""
import json
import math
import re
from collections import defaultdict

from langchain_core.documents import Document

from .config import CHROMA_DIR, COLLECTION_NAME, CORPUS_PATH
from .embeddings import get_embedding_function, embedding_is_semantic
import chromadb

_client = None
_col = None
_corpus = None
_bm25 = None


def _get_collection():
    global _client, _col
    if _col is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _col = _client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=get_embedding_function()
        )
    return _col


def _load_corpus_map():
    global _corpus
    if _corpus is None:
        m = {}
        with open(CORPUS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                m[o["id"]] = (o.get("title", ""), o["text"])
        _corpus = m
    return _corpus


def _bigrams(s: str):
    s = re.sub(r"\s+", "", s)
    return [s[i : i + 2] for i in range(len(s) - 1)]


class BM25:
    """中文二元组 BM25（标题加权 2x）。纯 Python，零依赖。"""

    def __init__(self, docs):
        self.ids = [d[0] for d in docs]
        tokenized, df, lengths = {}, defaultdict(int), []
        for id_, title, text in docs:
            toks = _bigrams(title) * 2 + _bigrams(text)  # 标题权重更高
            tokenized[id_] = toks
            lengths.append(len(toks))
            for t in set(toks):
                df[t] += 1
        self.tokenized = tokenized
        self.df = df
        self.N = len(docs)
        self.avgdl = sum(lengths) / max(1, len(lengths))
        self.k1, self.b = 1.5, 0.75

    def _idf(self, t):
        return math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))

    def rank(self, query):
        qt = _bigrams(query)
        if not qt:
            return []
        scored = []
        for id_ in self.ids:
            toks = self.tokenized[id_]
            dl = len(toks)
            freq = defaultdict(int)
            for t in toks:
                freq[t] += 1
            s = 0.0
            for t in qt:
                if t not in self.df:
                    continue
                ft = freq[t]
                if ft == 0:
                    continue
                s += self._idf(t) * (ft * (self.k1 + 1)) / (
                    ft + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            if s > 0:
                scored.append((id_, s))
        scored.sort(key=lambda x: -x[1])
        return scored


def _get_bm25():
    global _bm25
    if _bm25 is None:
        corpus = _load_corpus_map()
        docs = [(id_, t, x) for id_, (t, x) in corpus.items()]
        _bm25 = BM25(docs)
    return _bm25


def retrieve(question: str, k: int = 5):
    """返回 top-k 最相关文档（LangChain Document）。

    检索策略：
    - 真·语义编码（ONNX MiniLM）可用时：向量召回 + BM25 召回做 RRF 融合；
    - 降级为词法哈希向量时：向量与 BM25 本质同源（都是词法），改用纯 BM25，
      避免冗余融合拖累中文召回（BM25 在中文二元组下 Recall@5≈0.95）。
    """
    corpus = _load_corpus_map()
    bm25_ranked = _get_bm25().rank(question)

    if embedding_is_semantic():
        col = _get_collection()
        n = min(max(k * 3, 15), col.count())
        res = col.query(query_texts=[question], n_results=n)
        vec_ids = res["ids"][0]
        rrf = {}
        for rank, id_ in enumerate(vec_ids):
            rrf[id_] = rrf.get(id_, 0.0) + 1.0 / (rank + 60)
        for rank, (id_, _sc) in enumerate(bm25_ranked):
            rrf[id_] = rrf.get(id_, 0.0) + 1.0 / (rank + 60)
    else:
        rrf = {id_: 1.0 / (rank + 60) for rank, (id_, _sc) in enumerate(bm25_ranked)}

    docs = []
    for id_, score in sorted(rrf.items(), key=lambda x: -x[1])[:k]:
        title, text = corpus[id_]
        docs.append(
            Document(
                page_content=text,
                metadata={"id": id_, "title": title, "hybrid_score": round(score, 4)},
            )
        )
    return docs


def format_docs(docs):
    """把文档拼成可供大模型阅读的上下文块。"""
    return "\n\n".join(
        f"[来源 {d.metadata['id']} | {d.metadata['title']}]\n{d.page_content}" for d in docs
    )
