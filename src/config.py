"""全局配置：路径、集合名、LLM 接入开关。

LLM 接入遵循 OpenAI 兼容协议。未配置 KB_LLM_BASE_URL 时，
系统进入「离线演示模式」——仅做检索增强（不调用生成式大模型），
用确定性摘要验证「检索→接地」链路本身，不依赖外部服务。
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
CHROMA_DIR = BASE_DIR / "chroma_store"

# chromadb 集合名规则：3~512 字符，且必须以 [a-zA-Z0-9] 开头/结尾
COLLECTION_NAME = "teamflow_kb"

# —— LLM 接入（OpenAI 兼容）——
# 留空 => 离线演示模式（mock）。填入你自建网关或任意 OpenAI 兼容端点即启用真实生成。
LLM_BASE_URL = os.getenv("KB_LLM_BASE_URL", "")   # 例如 http://127.0.0.1:8787/v1
LLM_API_KEY = os.getenv("KB_LLM_API_KEY", "local")
LLM_MODEL = os.getenv("KB_LLM_MODEL", "deepseek-v4-flash")
