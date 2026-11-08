"""可插拔向量编码器：优先用 chromadb 内置 ONNX MiniLM-L6-v2（真·语义），
无 onnxruntime（或加载失败，如受限沙箱）时自动降级为纯 Python 二元组哈希向量
（零依赖、中文词法友好），保证开箱即跑、无环境依赖也能端到端验证。

两种编码器都输出定长稠密向量，入库与查询必须同源，避免向量错位。
"""
import hashlib
import re

import chromadb


class BagOfBigramsEmbedding:
    """零依赖的字符二元组词袋向量（TF 归一化）。对中文词法高度敏感。"""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed_one(self, text: str):
        vec = [0.0] * self.dim
        s = re.sub(r"\s+", "", text)
        for i in range(len(s) - 1):
            h = int(hashlib.md5(s[i : i + 2].encode("utf-8")).hexdigest(), 16) % self.dim
            vec[h] += 1.0
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def __call__(self, input):
        if isinstance(input, str):
            return self._embed_one(input)
        return [self._embed_one(t) for t in input]


def get_embedding_function():
    """返回可用的编码器；ONNX MiniLM 可用则用，否则降级为纯 Python 实现。"""
    try:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        ef = ONNXMiniLM_L6_V2()
        ef(["__probe__"])  # 触发模型加载；失败则降级
        _SEMANTIC[0] = True
        return ef
    except Exception:
        _SEMANTIC[0] = False
        return BagOfBigramsEmbedding()


# 标记当前编码器是否为「真·语义」模型（ONNX MiniLM）。
# 非语义（纯 Python 词法哈希）时，检索改用纯 BM25，避免与词法向量冗余融合。
_SEMANTIC = [False]


def embedding_is_semantic() -> bool:
    return _SEMANTIC[0]
