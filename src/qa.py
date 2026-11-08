"""问答入口：统一 ask()，自动选择真实大模型或离线演示。

对外只暴露 ask(question, k)，调用方无需关心底层是 mock 还是真模型。
返回 {answer, sources, mode}，sources 给出可溯源的文档清单。
"""
from .retriever import retrieve, format_docs
from .llm import mock_answer, use_real_llm
from .chain import build_chain


def ask(question: str, k: int = 5):
    docs = retrieve(question, k=k)
    if use_real_llm():
        chain = build_chain()
        context = format_docs(docs)
        answer = chain.invoke({"question": question, "context": context})
        mode = "llm"
    else:
        answer = mock_answer(question, docs)
        mode = "mock"
    sources = [
        {"id": d.metadata["id"], "title": d.metadata["title"]} for d in docs
    ]
    return {"answer": answer, "sources": sources, "mode": mode}


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "忘记密码怎么重置？"
    r = ask(q)
    print(f"[模式] {r['mode']}")
    print(f"[问题] {q}")
    print("[回答]")
    print(r["answer"])
    print("[来源]", [(s["id"], s["title"]) for s in r["sources"]])
