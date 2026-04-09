"""OpenCortex — HTTP API (FastAPI)."""
from __future__ import annotations

import json
import os
import posixpath
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import unquote

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
    WIKI_DIRNAME,
    _bundle_artifacts_summary,
    ask_stream as rag_ask_stream,
    list_kbs,
    load_search_bundle,
    search_bundle_artifact_signature,
)
from wiki import is_wiki_write_in_progress, save_query_note

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
_graph_report_snapshot: dict[str, Any] | None = None
_cfg: dict[str, str] = {}
_SEARCH_BUNDLE_RELOAD_RETRIES = 3
_SEARCH_BUNDLE_RELOAD_RETRY_DELAY_SECONDS = 0.05


def _search_artifact_signature(persist_dir: str) -> tuple[tuple[str, bool, int, int], ...]:
    return search_bundle_artifact_signature(persist_dir)


def _wiki_write_in_progress(persist_dir: str) -> bool:
    try:
        return is_wiki_write_in_progress(Path(persist_dir))
    except OSError:
        return False


def _read_graph_report_snapshot(manifest: dict[str, Any], persist_dir: str) -> dict[str, Any]:
    relpath = str(
        manifest.get("graph_report_file", f"{REPORTS_DIRNAME}/{GRAPH_REPORT_FILENAME}")
        or f"{REPORTS_DIRNAME}/{GRAPH_REPORT_FILENAME}"
    )
    report_path = Path(persist_dir) / Path(relpath)
    try:
        content = report_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"path": relpath, "content": None, "exists": False}
    return {"path": relpath, "content": content, "exists": True}


def _refresh_search_bundle_if_needed(force: bool = False):
    global _search_bundle, _search_bundle_signature, _graph_report_snapshot
    current_signature = _search_artifact_signature(_cfg["persist_dir"])
    if not force and _search_bundle is not None and current_signature == _search_bundle_signature:
        return _search_bundle

    previous_bundle = _search_bundle
    last_error: RuntimeError | None = None
    for attempt in range(_SEARCH_BUNDLE_RELOAD_RETRIES):
        if _wiki_write_in_progress(_cfg["persist_dir"]):
            last_error = RuntimeError("wiki 页面正在重建，请稍后重试。")
            if attempt < _SEARCH_BUNDLE_RELOAD_RETRIES - 1:
                time.sleep(_SEARCH_BUNDLE_RELOAD_RETRY_DELAY_SECONDS)
                continue
            break
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
        if _wiki_write_in_progress(_cfg["persist_dir"]):
            last_error = RuntimeError("wiki 页面正在重建，请稍后重试。")
            if attempt < _SEARCH_BUNDLE_RELOAD_RETRIES - 1:
                time.sleep(_SEARCH_BUNDLE_RELOAD_RETRY_DELAY_SECONDS)
                continue
            break
        after_signature = _search_artifact_signature(_cfg["persist_dir"])
        if before_signature == after_signature:
            try:
                graph_report_snapshot = _read_graph_report_snapshot(bundle.manifest, _cfg["persist_dir"])
            except OSError as exc:
                last_error = RuntimeError(f"结构报告快照读取失败：{exc}")
                if attempt < _SEARCH_BUNDLE_RELOAD_RETRIES - 1:
                    time.sleep(_SEARCH_BUNDLE_RELOAD_RETRY_DELAY_SECONDS)
                    continue
                break
            final_signature = _search_artifact_signature(_cfg["persist_dir"])
            if after_signature == final_signature:
                _search_bundle = bundle
                _search_bundle_signature = final_signature
                _graph_report_snapshot = graph_report_snapshot
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


class SaveQueryRequest(BaseModel):
    question: str
    answer: str
    sources: list[dict[str, Any]]
    tags: list[str] | None = None


def _wiki_page_summary(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(page.get("id", "") or ""),
        "kind": str(page.get("kind", "") or ""),
        "title": str(page.get("title", "") or ""),
        "relpath": str(page.get("relpath", "") or ""),
        "source_refs": [
            str(source)
            for source in page.get("source_refs", [])
            if isinstance(source, str) and source
        ],
    }


def _wiki_page_payload(page: dict[str, Any]) -> dict[str, Any]:
    payload = _wiki_page_summary(page)
    payload["content"] = str(page.get("text", "") or "")
    return payload


def _normalize_wiki_page_relpath(page_path: str) -> str:
    raw = unquote(str(page_path or "").strip()).replace("\\", "/")
    if not raw or raw == ".":
        return "index.md"
    normalized = posixpath.normpath(raw).lstrip("/")
    if normalized.startswith(f"{WIKI_DIRNAME}/"):
        normalized = normalized[len(WIKI_DIRNAME) + 1 :]
    if normalized in {"", "."}:
        return "index.md"
    if normalized == ".." or normalized.startswith("../"):
        raise HTTPException(status_code=400, detail="wiki 页面路径非法")
    return normalized


def _find_wiki_page(search_bundle, relpath: str) -> dict[str, Any] | None:
    normalized_relpath = _normalize_wiki_page_relpath(relpath)
    for page in search_bundle.wiki_pages:
        if not isinstance(page, dict):
            continue
        if str(page.get("relpath", "") or "") == normalized_relpath:
            return page
    return None


def _graph_report_relpath(search_bundle) -> str:
    return str(
        search_bundle.manifest.get("graph_report_file", f"{REPORTS_DIRNAME}/{GRAPH_REPORT_FILENAME}")
        or f"{REPORTS_DIRNAME}/{GRAPH_REPORT_FILENAME}"
    )


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
        if result.get("wiki_trace") is not None:
            yield {
                "event": "wiki_trace",
                "data": json.dumps(result["wiki_trace"], ensure_ascii=False),
            }
        if result.get("artifacts") is not None:
            yield {
                "event": "artifacts",
                "data": json.dumps(result["artifacts"], ensure_ascii=False),
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
    if req.debug and result.get("artifacts") is None:
        result["artifacts"] = _bundle_artifacts_summary(search_bundle)

    if not req.stream:
        answer = "".join(result["answer_stream"])
        payload = {"answer": answer, "sources": result["sources"]}
        if req.debug:
            payload["bridge_entities"] = result.get("bridge_entities", [])
            payload["wiki_trace"] = result.get("wiki_trace", [])
            payload["artifacts"] = result.get("artifacts", {})
            payload["search_trace"] = result.get("search_trace", [])
        return payload

    return _stream_response(result)


@app.get("/api/wiki/index")
def wiki_index():
    search_bundle = _refresh_search_bundle_if_needed()
    artifacts = _bundle_artifacts_summary(search_bundle)
    counts: dict[str, int] = {}
    for page in search_bundle.wiki_pages:
        if not isinstance(page, dict):
            continue
        kind = str(page.get("kind", "") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1

    index_page = _find_wiki_page(search_bundle, "index.md")
    return {
        "index": _wiki_page_payload(index_page) if isinstance(index_page, dict) else None,
        "pages": artifacts["wiki_pages"],
        "counts": counts,
        "artifacts": artifacts,
    }


@app.get("/api/wiki/page/{page_path:path}")
def wiki_page(page_path: str):
    search_bundle = _refresh_search_bundle_if_needed()
    page = _find_wiki_page(search_bundle, page_path)
    if page is None:
        raise HTTPException(status_code=404, detail="wiki 页面不存在")
    return {"page": _wiki_page_payload(page)}


@app.post("/api/wiki/save-query")
def save_query(req: SaveQueryRequest):
    try:
        result = save_query_note(
            persist_path=Path(_cfg["persist_dir"]),
            question=req.question,
            answer=req.answer,
            sources=req.sources,
            tags=req.tags,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"保存 query note 失败：{exc}") from exc

    search_bundle = _refresh_search_bundle_if_needed()
    note_page = _find_wiki_page(search_bundle, result["note_relpath"])
    return {
        "note": result,
        "page": _wiki_page_payload(note_page) if isinstance(note_page, dict) else None,
        "artifacts": _bundle_artifacts_summary(search_bundle),
    }


@app.get("/api/graph/report")
def graph_report():
    search_bundle = _refresh_search_bundle_if_needed()
    snapshot = _graph_report_snapshot
    relpath = _graph_report_relpath(search_bundle)
    if not isinstance(snapshot, dict):
        raise HTTPException(status_code=503, detail="结构报告快照尚未就绪，请稍后重试。")
    if not bool(snapshot.get("exists")):
        raise HTTPException(status_code=404, detail="结构报告不存在")
    content = str(snapshot.get("content", "") or "")
    return {
        "path": str(snapshot.get("path", "") or relpath),
        "content": content,
        "artifacts": _bundle_artifacts_summary(search_bundle),
    }
