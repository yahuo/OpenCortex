"""OpenCortex — HTTP API (FastAPI)."""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ragbot import (
    COMMUNITY_INDEX_FILENAME,
    DOCUMENT_GRAPH_FILENAME,
    ENTITY_GRAPH_FILENAME,
    GRAPH_REPORT_FILENAME,
    LINT_REPORT_FILENAME,
    REPORTS_DIRNAME,
    SEARCH_MODES,
    ask_stream as rag_ask_stream,
    list_kbs,
    load_search_bundle,
)

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
_search_bundle_signature: tuple[tuple[str, bool, int, int], ...] | None = None
_cfg: dict[str, str] = {}
_SEARCH_BUNDLE_RELOAD_RETRIES = 3
_SEARCH_BUNDLE_RELOAD_RETRY_DELAY_SECONDS = 0.05


def _search_artifact_signature(persist_dir: str) -> tuple[tuple[str, bool, int, int], ...]:
    persist_path = Path(persist_dir)
    manifest_path = persist_path / "index_manifest.json"
    manifest: dict[str, str] = {}
    if manifest_path.exists():
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded_manifest, dict):
                manifest = loaded_manifest
        except (OSError, json.JSONDecodeError):
            manifest = {}

    relative_paths = [
        "index.faiss",
        "index.pkl",
        "index_manifest.json",
        str(manifest.get("document_graph_file", DOCUMENT_GRAPH_FILENAME) or DOCUMENT_GRAPH_FILENAME),
        str(manifest.get("entity_graph_file", ENTITY_GRAPH_FILENAME) or ENTITY_GRAPH_FILENAME),
        str(manifest.get("community_index_file", COMMUNITY_INDEX_FILENAME) or COMMUNITY_INDEX_FILENAME),
        str(manifest.get("lint_report_file", LINT_REPORT_FILENAME) or LINT_REPORT_FILENAME),
        str(
            manifest.get("graph_report_file", f"{REPORTS_DIRNAME}/{GRAPH_REPORT_FILENAME}")
            or f"{REPORTS_DIRNAME}/{GRAPH_REPORT_FILENAME}"
        ),
    ]

    signature: list[tuple[str, bool, int, int]] = []
    seen: set[str] = set()
    for relative_path in relative_paths:
        normalized = Path(relative_path).as_posix()
        if normalized in seen:
            continue
        seen.add(normalized)
        artifact_path = persist_path / normalized
        try:
            stat = artifact_path.stat()
        except OSError:
            signature.append((normalized, False, 0, 0))
            continue
        signature.append((normalized, True, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def _refresh_search_bundle_if_needed(force: bool = False):
    global _search_bundle, _search_bundle_signature
    current_signature = _search_artifact_signature(_cfg["persist_dir"])
    if not force and _search_bundle is not None and current_signature == _search_bundle_signature:
        return _search_bundle

    previous_bundle = _search_bundle
    last_error: RuntimeError | None = None
    for attempt in range(_SEARCH_BUNDLE_RELOAD_RETRIES):
        before_signature = _search_artifact_signature(_cfg["persist_dir"])
        try:
            bundle = load_search_bundle(
                embed_api_key=_cfg["embed_api_key"],
                embed_base_url=_cfg["embed_base_url"],
                embed_model=_cfg["embed_model"],
                persist_dir=_cfg["persist_dir"],
            )
        except Exception as exc:
            last_error = RuntimeError(f"检索索引重载失败：{exc}")
            if attempt < _SEARCH_BUNDLE_RELOAD_RETRIES - 1:
                time.sleep(_SEARCH_BUNDLE_RELOAD_RETRY_DELAY_SECONDS)
                continue
            break
        if bundle is None:
            last_error = RuntimeError("检索索引加载失败，请先运行 start.py 重建索引。")
            break
        after_signature = _search_artifact_signature(_cfg["persist_dir"])
        if before_signature == after_signature:
            _search_bundle = bundle
            _search_bundle_signature = after_signature
            return bundle
        if attempt < _SEARCH_BUNDLE_RELOAD_RETRIES - 1:
            time.sleep(_SEARCH_BUNDLE_RELOAD_RETRY_DELAY_SECONDS)

    if not force and previous_bundle is not None:
        return previous_bundle
    raise last_error or RuntimeError("检索索引正在更新，请稍后重试。")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cfg
    _cfg = _read_config()
    _refresh_search_bundle_if_needed(force=True)
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
        if result.get("bridge_entities") is not None:
            yield {
                "event": "bridge_entities",
                "data": json.dumps(result["bridge_entities"], ensure_ascii=False),
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

    search_bundle = _refresh_search_bundle_if_needed()
    result = rag_ask_stream(
        question=req.question,
        search_bundle=search_bundle,
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
            payload["bridge_entities"] = result.get("bridge_entities", [])
            payload["search_trace"] = result.get("search_trace", [])
        return payload

    return _stream_response(result)
