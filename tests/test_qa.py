"""离线冒烟测试：入库 + 检索 + 离线摘要，不依赖任何外部大模型。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import build_vectorstore
from src.qa import ask


def test_retrieval_and_mock_answer():
    build_vectorstore()
    res = ask("TeamFlow 支持哪些第三方集成？", k=5)
    assert res["sources"], "应返回来源列表"
    ids = [s["id"] for s in res["sources"]]
    assert "kb005" in ids, f"期望检索到 kb005，实际：{ids}"
    assert res["answer"] and res["mode"] == "mock"
    print("PASS retrieval+answer, top sources:", ids[:3])


if __name__ == "__main__":
    test_retrieval_and_mock_answer()
