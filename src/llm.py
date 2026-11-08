"""LLM 接入层：真实大模型（OpenAI 兼容）与离线演示（mock）两条路径。

mock 不调用任何外部模型，仅基于检索结果做确定性摘要，
用于在没有 API key / 网关的环境下证明「检索→接地」链路可跑通。
"""
import re

from .config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL


def use_real_llm() -> bool:
    return bool(LLM_BASE_URL.strip())


def get_llm():
    """返回 LangChain ChatOpenAI（OpenAI 兼容）。未配置则返回 None。"""
    if not use_real_llm():
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.3,
    )


def _best_sentence(question: str, text: str) -> str:
    """从一段文本中挑出与问题字符重叠最高的句子（离线摘要用）。"""
    q_chars = set(re.sub(r"\s+", "", question))
    sentences = re.split(r"[。！？\n]", text)
    best, best_score = "", -1.0
    for s in sentences:
        s = s.strip()
        if len(s) < 4:
            continue
        overlap = len(q_chars & set(s))
        score = overlap / max(1, len(s))
        if score > best_score:
            best, best_score = s, score
    return best or text[:60]


def mock_answer(question: str, docs) -> str:
    """离线演示回答：列出检索到的来源 + 每个来源最相关句。"""
    if not docs:
        return "（离线演示模式）知识库未检索到相关条目，请换个问法或先执行入库。"
    lines = [
        "（离线演示模式 · 仅做检索增强，未调用生成式大模型）",
        f"问题：{question}",
        "根据知识库检索，最相关的条目如下：",
    ]
    for d in docs[:3]:
        snippet = _best_sentence(question, d.page_content)
        lines.append(f"· [{d.metadata['id']} {d.metadata['title']}] {snippet}")
    lines.append("—— 配置 KB_LLM_BASE_URL 后，以上检索结果将作为上下文生成自然回答。")
    return "\n".join(lines)
