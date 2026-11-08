"""入库：把语料写进 chromadb 持久化集合。

编码器可插拔（见 src/embeddings）：默认 chromadb 内置 ONNX MiniLM-L6-v2
（需 onnxruntime），否则自动降级为纯 Python 二元组哈希向量（零依赖）。
如需中文优化 embedding，把 get_embedding_function() 换成
sentence_transformers 的 BAAI/bge-small-zh-v1.5 即可，其余代码一字不改。
"""
import json
from pathlib import Path

import chromadb

from .config import CORPUS_PATH, CHROMA_DIR, COLLECTION_NAME
from .embeddings import get_embedding_function


def load_corpus(path=CORPUS_PATH):
    """读取 JSONL 语料，每行 {id, title, text}。"""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def build_vectorstore(path=CHROMA_DIR, collection_name=COLLECTION_NAME, corpus_path=CORPUS_PATH):
    """（重建）向量库：删除旧集合后重新写入，保证可重复构建。"""
    client = chromadb.PersistentClient(path=str(path))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.get_or_create_collection(
        name=collection_name, embedding_function=get_embedding_function()
    )

    corpus = load_corpus(corpus_path)
    ids = [d["id"] for d in corpus]
    texts = [d["text"] for d in corpus]
    metadatas = [{"id": d["id"], "title": d.get("title", "")} for d in corpus]
    col.add(ids=ids, documents=texts, metadatas=metadatas)
    return col


if __name__ == "__main__":
    c = build_vectorstore()
    print("入库完成，文档数：", c.count())
