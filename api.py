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

from ragbot import SEARCH_MODES, ask_stream as rag_ask_stream
from ragbot import list_kbs, load_search_bundle

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
        "search_mode": os.getenv("SEARCH_MODE", "hybrid").strip().lower() or "hybrid",
    }


_search_bundle = None
_cfg: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _search_bundle, _cfg
    _cfg = _read_config()
    _search_bundle = load_search_bundle(
        embed_api_key=_cfg["embed_api_key"],
        embed_base_url=_cfg["embed_base_url"],
        embed_model=_cfg["embed_model"],
        persist_dir=_cfg["persist_dir"],
    )
    if _search_bundle is None:
        raise RuntimeError("检索索引加载失败，请先运行 start.py 重建索引。")
    yield


app = FastAPI(title="OpenCortex API", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    stream: bool = False
    kb: str | None = None
    search_mode: str | None = None
    debug: bool = False


def _stream_response(result: dict):
    def event_generator():
        for chunk in result["answer_stream"]:
            yield {"event": "chunk", "data": chunk}
        yield {
            "event": "sources",
            "data": json.dumps(result["sources"], ensure_ascii=False),
        }
        if result.get("search_trace") is not None:
            yield {
                "event": "search_trace",
                "data": json.dumps(result["search_trace"], ensure_ascii=False),
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

    search_mode = (req.search_mode or _cfg["search_mode"]).strip().lower()
    if search_mode not in SEARCH_MODES:
        raise HTTPException(
            status_code=400,
            detail="search_mode 必须是 vector、hybrid 或 agentic",
        )

    if req.kb is not None:
        available = list_kbs(persist_dir=_cfg["persist_dir"])
        if req.kb not in available:
            raise HTTPException(
                status_code=400,
                detail=f"知识库 '{req.kb}' 不存在，可用知识库: {available}",
            )

    result = rag_ask_stream(
        question=req.question,
        search_bundle=_search_bundle,
        llm_api_key=_cfg["llm_api_key"],
        llm_model=_cfg["llm_model"],
        llm_base_url=_cfg["llm_base_url"],
        kb=req.kb,
        search_mode=search_mode,
        debug=req.debug,
    )

    if not req.stream:
        answer = "".join(result["answer_stream"])
        payload = {"answer": answer, "sources": result["sources"]}
        if req.debug:
            payload["search_trace"] = result.get("search_trace", [])
        return payload

    return _stream_response(result)
