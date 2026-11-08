# langchain-kb-qa（部署版）

基于 **LangChain + chromadb** 的检索增强问答（RAG）最小可部署系统。
中文垂域语料实跑，默认**离线演示模式**（仅检索增强，不调用生成式大模型），
配置一个 OpenAI 兼容端点即可切换为真实大模型生成。

> 语料为**虚构的 SaaS 帮助中心**，不含任何真实业务数据。

## 架构

```
用户问题
   │
   ▼
[检索层]  chromadb 向量召回 ┐
                          ├─ RRF 融合（真·语义编码时）/ 纯 BM25（降级时）
        中文二元组 BM25 召回 ┘
   │  top-k 相关文档（LangChain Document）
   ▼
[生成层]  LangChain LCEL 链：prompt → LLM → StrOutputParser
   │  （离线模式：确定性检索摘要，不调用模型）
   ▼
{ answer, sources }  →  FastAPI /api/ask  →  前端聊天页
```

- **检索层（核心）**：向量召回 + 中文二元组 BM25 重排。BM25 在中文垂域召回极强，
  与向量互补；两个引擎结果集重叠低，正是混合检索的价值所在。
- **生成层（可插拔）**：默认 mock；填 `KB_LLM_BASE_URL` 即用任意 OpenAI 兼容大模型。
- **部署层**：FastAPI + 单页前端 + Dockerfile，一条命令起服务。

## 编码器与检索策略（可插拔、自动降级）

编码器由 `src/embeddings.py` 统一提供，**入库与查询同源**，避免向量错位：

| 环境 | 编码器 | 检索策略 |
|---|---|---|
| 装了 `onnxruntime` | chromadb 内置 **ONNX MiniLM-L6-v2**（384d，真·语义） | 向量召回 + BM25 → **RRF 融合** |
| 无 `onnxruntime`（含受限沙箱） | 纯 Python **二元组哈希向量**（零依赖、词法） | **纯 BM25**（词法向量与 BM25 同源，融合反而稀释召回） |

降级是有意为之：受限环境里 `onnxruntime` 会触发 OpenBLAS 内存分配失败，
自动降级保证「开箱即跑、可端到端验证」。 BM25 在中文二元组分词下
Recall@5≈0.95（见姊妹项目 [rag-vector-vs-keyword](https://github.com/Babymrbbbb/rag-vector-vs-keyword)），
降级后中文召回反而更稳。

**中文语义升级（可选）**：把 `get_embedding_function()` 换成
`sentence-transformers` 的 `BAAI/bge-small-zh-v1.5` 即可，其余代码不变。
英文预训练的 MiniLM 在中文垂域召回有限，中文优化 embedding 才是生产解法。

## 快速开始

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 1) 构建向量库（首次或语料变更后）
python -m src.ingest

# 2) 启动服务（自动确保向量库存在）
uvicorn app:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

离线模式也能跑通完整链路：检索 → 接地摘要 → 返回可溯源来源。

### 命令行直测

```bash
python -m src.qa "忘记密码怎么重置？"   # 直接看回答 + 来源
python tests/test_qa.py                 # 离线冒烟测试（入库 + 检索 + 摘要）
```

## 接入真实大模型

设置环境变量后，`src/chain.build_chain()` 自动启用 LangChain `prompt | llm | parser`：

```bash
set KB_LLM_BASE_URL=http://127.0.0.1:8787/v1   # 或 https://api.openai.com/v1
set KB_LLM_API_KEY=sk-xxx
set KB_LLM_MODEL=deepseek-v4-flash
uvicorn app:app --port 8000
```

前端「模式」标签会显示「真实大模型」。prompt 已约束「只依据知识库、不编造」。

## 部署（Docker）

```bash
docker build -t teamflow-kbqa .
docker run -p 8000:8000 teamflow-kbqa
```

## 目录结构

```
langchain-kb-qa/
├── data/corpus.jsonl        # 虚构 SaaS 帮助中心语料（20 条）
├── src/
│   ├── config.py            # 路径 / 集合名 / LLM 开关
│   ├── embeddings.py        # 可插拔编码器（ONNX MiniLM / 纯 Python 降级）
│   ├── ingest.py            # 入库 chromadb
│   ├── retriever.py         # 向量召回 + BM25 重排（RRF）→ LangChain Document
│   ├── llm.py               # 真实大模型 / 离线 mock 双路径
│   ├── chain.py             # LangChain LCEL 生产链路
│   └── qa.py                # ask() 统一入口 + 命令行直测
├── app.py                   # FastAPI 服务
├── static/index.html        # 单页聊天 UI
├── tests/test_qa.py         # 离线冒烟测试
├── Dockerfile
└── requirements.txt
```

## 边界与说明

- 离线模式**不调用生成式大模型**，仅证明「检索 → 接地 → 可溯源」链路可工程化落地；
  真实自然语言回答由接入的大模型负责。
- 检索质量是 RAG 的上限：本项目把检索层做扎实（向量 + BM25 混合、中文友好、可升级），
  生成层保持可插拔，避免把「幻觉」包装成能力。
- 语料为虚构演示数据，仅用于验证检索与工程链路。
- 已在 Python 3.13 / chromadb 环境端到端实跑：离线冒烟测试通过、
  FastAPI `/api/ask` 返回 200、6 组中文问题的正确文档均排在首位。
