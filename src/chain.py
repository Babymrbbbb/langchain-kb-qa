"""生产链路：LangChain LCEL 组装检索增强问答链（真实大模型路径）。

离线演示路径见 qa.ask()（默认走 mock），本文件提供真实大模型可用的
prompt | llm | parser 标准三段式，供部署时启用。
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .llm import get_llm
from .retriever import format_docs

SYSTEM_PROMPT = (
    "你是 TeamFlow 企业协作平台的智能客服助手。"
    "只依据下方【知识库内容】作答，不得编造；"
    "若知识库没有相关信息，明确说明未找到，不要臆测。"
)


def build_chain():
    """组装 RetrievalQA 风格的 LCEL 链：prompt -> llm -> 文本解析。"""
    llm = get_llm()
    if llm is None:
        raise RuntimeError(
            "未配置 KB_LLM_BASE_URL，无法构建真实大模型链路；"
            "请使用 qa.ask() 的离线演示模式，或先配置环境变量。"
        )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "用户问题：{question}\n\n【知识库内容】\n{context}"),
        ]
    )
    return prompt | llm | StrOutputParser()
