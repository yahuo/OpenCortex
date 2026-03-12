"""OpenCortex — HTTP API (FastAPI)."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ragbot import ask_stream as rag_ask_stream
from ragbot import list_kbs, load_vectorstore

load_dotenv()


def _read_config() -> dict[str, str]:
    return {
        "persist_dir": str(
            Path(os.getenv("CHROMA_PERSIST_DIR", "~/wechat_rag_db")).expanduser()
        ),
        "embed_api_key": os.getenv("EMBED_API_KEY", "").strip(),
        "embed_base_url": os.getenv(
            "EMBED_BASE_URL", "https://api.siliconflow.cn/v1"
        ).strip(),
        "embed_model": os.getenv("EMBED_MODEL", "BAAI/bge-m3").strip(),
        "llm_api_key": os.getenv("LLM_API_KEY", "").strip(),
        "llm_base_url": os.getenv(
            "LLM_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ).strip(),
        "llm_model": os.getenv("LLM_MODEL", "gemini-2.0-flash").strip(),
    }


_vectorstore = None
_cfg: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _vectorstore, _cfg
    _cfg = _read_config()
    _vectorstore = load_vectorstore(
        embed_api_key=_cfg["embed_api_key"],
        embed_base_url=_cfg["embed_base_url"],
        embed_model=_cfg["embed_model"],
        persist_dir=_cfg["persist_dir"],
    )
    if _vectorstore is None:
        raise RuntimeError("向量索引加载失败，请先运行 start.py 重建索引。")
    yield


app = FastAPI(title="OpenCortex API", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    stream: bool = False
    kb: str | None = None


def _stream_response(result: dict):
    def event_generator():
        for chunk in result["answer_stream"]:
            yield {"event": "chunk", "data": chunk}
        yield {
            "event": "sources",
            "data": json.dumps(result["sources"], ensure_ascii=False),
        }
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/kbs")
def kbs():
    return {"kbs": list_kbs(persist_dir=_cfg["persist_dir"])}


@app.post("/api/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")

    if req.kb is not None:
        available = list_kbs(persist_dir=_cfg["persist_dir"])
        if req.kb not in available:
            raise HTTPException(
                status_code=400,
                detail=f"知识库 '{req.kb}' 不存在，可用知识库: {available}",
            )

    result = rag_ask_stream(
        question=req.question,
        vectorstore=_vectorstore,
        llm_api_key=_cfg["llm_api_key"],
        llm_model=_cfg["llm_model"],
        llm_base_url=_cfg["llm_base_url"],
        kb=req.kb,
    )

    if not req.stream:
        answer = "".join(result["answer_stream"])
        return {"answer": answer, "sources": result["sources"]}

    return _stream_response(result)
