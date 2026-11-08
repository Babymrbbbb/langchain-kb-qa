"""部署入口：FastAPI 提供 /api/ask 与静态首页，可直接容器化部署。

启动：uvicorn app:app --host 0.0.0.0 --port 8000
默认离线演示模式（检索增强摘要）；配置 KB_LLM_BASE_URL 后自动切换真实生成。
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.ingest import build_vectorstore
from src.qa import ask
from src.retriever import _get_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动即确保向量库存在（首次会自动构建）
    try:
        _get_collection()
    except Exception:
        build_vectorstore()
    yield


app = FastAPI(title="TeamFlow KB-QA", version="1.0.0", lifespan=lifespan)


class AskReq(BaseModel):
    question: str
    k: int = 5


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.post("/api/ask")
def api_ask(req: AskReq):
    if not req.question.strip():
        return JSONResponse({"error": "question required"}, status_code=400)
    return ask(req.question, k=req.k)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
