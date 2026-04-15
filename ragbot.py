#!/usr/bin/env python3
"""
ragbot.py — 通用本地文档 RAG / Hybrid Search 核心模块

架构：
  OpenAI 兼容 Embedding API → FAISS (向量存储)
  + 本地 text cache / symbol index / hybrid retrieval
  → 任意 OpenAI 兼容 LLM (生成)
"""
from __future__ import annotations

from collections import deque
import fnmatch
import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragbot_artifacts import (
    current_build_config_snapshot,
    current_build_snapshot,
    current_source_snapshot,
    source_snapshot_from_indexed_files,
)
from ragbot_build import (
    _build_document_graph,
    _clear_generated_wiki_artifacts,
    _write_index_artifacts,
    build_vectorstore,
)
from ragbot_chunking import (
    _chunk_markdown_by_heading,
    _chunk_python_code,
    _chunk_structured_text,
    _extract_python_symbols,
    _iter_chunk_specs_from_line_stream,
    _iter_chunks_from_cached_file,
    _iter_markdown_heading_chunks_from_file,
    _iter_text_file_lines,
    _iter_wechat_markdown_chunks_from_file,
)
from ragbot_ingest import (
    _chunks_for_indexed_text,
    _convert_binary_to_markdown,
    _normalized_cache_path,
    _normalize_structured_text,
    _prepare_indexed_content,
    _process_source_file,
    _process_source_file_to_cache,
    _read_source_text,
)
from ragbot_sources import (
    _build_documents,
    _index_source_files,
    _iter_documents_for_indexed_files,
    _iter_supported_files,
    _should_ignore_relative_path,
)
from ragbot_pipeline import (
    _add_embedded_batch,
    _build_vectorstore_from_document_stream,
    _document_batch_ids,
    _embed_texts_with_retries,
    _iter_document_batches,
)
from ragbot_graph import (
    _build_community_index,
    _format_report_list,
    _project_file_relationships,
    _render_graph_report,
)
from ragbot_local_search import (
    _build_query_expansion_index,
    _build_fulltext_index,
    _bundle_sources,
    _extract_query_plan,
    _finalize_hits,
    _first_non_empty_lines,
    _fulltext_query_terms,
    _fulltext_terms,
    _grep_search_python,
    _grep_search_with_rg,
    _has_supported_file_suffix,
    _hit_priority,
    _is_extension_only_glob,
    _keyword_score,
    _looks_like_explicit_source_term,
    _merge_primary_kind,
    _read_cached_text,
    _search_max_steps,
    _search_mode,
    _symbol_match_score,
    ast_search,
    bm25_search,
    glob_search,
    grep_search,
)
from ragbot_notes import (
    _build_file_only_entity_graph,
    _entity_attachment_files,
    _entity_edge_evidence_source,
    _extract_markdown_section,
    _load_query_note_records,
    _merge_query_notes_into_entity_graph,
    _normalize_entity_edge_payload,
    _normalize_entity_node_id,
    _normalize_entity_node_payload,
    _parse_query_note_record,
    _query_notes_dir,
    _unescape_markdown_link_text,
    refresh_query_note_graph_artifacts,
)
from ragbot_retrieval import (
    _bundle_artifacts_summary,
    _call_retrieval_planner,
    _candidate_sources_from_hits,
    _collect_bridge_entities,
    _collect_wiki_trace,
    _document_graph_expand_candidate_sources,
    _expand_query_plan_from_corpus_index,
    _expand_query_plan_from_vector_feedback,
    _entity_graph_expand_candidate_sources,
    _expand_candidate_sources,
    _expand_candidate_sources_detailed,
    _extract_json_blob,
    _heuristic_expand_candidate_sources,
    _merge_grouped_hits,
    _merge_plans,
    _planner_prompt,
    _run_search_step,
    _should_stop_after_first_step,
    _sources_from_hits,
    _build_context_and_sources,
    _vector_search_raw,
    _vector_hit_from_document,
    _wiki_kind_priority,
    _wiki_query_terms,
    _wiki_source_ref_matches_query,
    _wiki_search,
    ask_stream,
    retrieve,
    vector_search,
)
from ragbot_semantic import (
    _build_entity_graph,
    _build_python_module_lookup,
    _build_symbol_reference_lookup,
    _entity_semantic_node_id,
    _extract_section_reference_tokens,
    _extract_semantic_sections,
    _load_semantic_cache,
    _parse_semantic_payload,
    _resolve_import_reference,
    _resolve_symbol_reference,
    _response_token_usage,
    _semantic_aliases,
    _semantic_cache_path,
    _semantic_extraction_prompt,
    _semantic_graph_enabled,
    _semantic_section_fingerprint,
    _upsert_semantic_node,
    _write_semantic_cache,
)
from ragbot_runtime import (
    SearchBundle,
    _load_wiki_pages,
    load_search_bundle,
    load_vectorstore,
    search_bundle_artifact_signature,
)

# ─────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
TIME_WINDOW_MINUTES = 30
DEFAULT_FAISS_DIR = str(Path.home() / "wechat_rag_db")
DEFAULT_TOP_K = 6
GENERIC_CHUNK_SIZE = 1200
GENERIC_CHUNK_OVERLAP_LINES = 3
DEFAULT_EMBED_BATCH_SIZE = 64
DEFAULT_EMBED_BATCH_SLEEP_SECONDS = 0.0
DEFAULT_SEARCH_MODE = "hybrid"
SEARCH_MODES = {"vector", "hybrid", "agentic"}
SEARCH_MAX_STEPS_DEFAULT = 2
NORMALIZED_TEXT_DIRNAME = "normalized_texts"
SYMBOL_INDEX_FILENAME = "symbol_index.jsonl"
DOCUMENT_GRAPH_FILENAME = "document_graph.json"
ENTITY_GRAPH_FILENAME = "entity_graph.json"
SEMANTIC_EXTRACT_CACHE_FILENAME = "semantic_extract_cache.json"
COMMUNITY_INDEX_FILENAME = "community_index.json"
REPORTS_DIRNAME = "reports"
GRAPH_REPORT_FILENAME = "GRAPH_REPORT.md"
LINT_REPORT_FILENAME = "lint_report.json"
WIKI_DIRNAME = "wiki"
QUERY_NOTES_DIRNAME = "queries"
PLANNER_TIMEOUT_SECONDS = 15
RRF_K = 60
RRF_WEIGHTS = {
    "ast": 1.0,
    "bm25": 0.85,
    "grep": 0.9,
    "glob": 0.6,
    "vector": 0.5,
}
GRAPH_MAX_NEIGHBORS = 8
GRAPH_SHARED_TOKEN_MIN_LENGTH = 3
GRAPH_SHARED_TOKEN_MAX_DOC_FREQ = 5
GRAPH_EDGE_PRIORITY = {
    "links_to": 0,
    "mentions_path": 1,
    "shared_symbol": 2,
    "same_series": 3,
}
ENTITY_EXPANSION_PRIORITY = {
    "references": 0,
    "imports": 1,
    "links_to": 2,
    "mentions_path": 3,
    "rationale_for": 4,
    "semantically_related": 5,
    "shared_symbol": 6,
    "same_series": 7,
}
ENTITY_INFERRED_EDGE_TYPES = {
    "shared_symbol",
    "same_series",
    "rationale_for",
    "semantically_related",
}
ENTITY_ATTACHMENT_NODE_TYPES = {"concept", "decision", "query_note"}
COMMUNITY_STRONG_EDGE_TYPES = {"links_to", "mentions_path", "references", "imports"}
COMMUNITY_TOP_FILES = 5
COMMUNITY_TOP_SYMBOLS = 5
COMMUNITY_TOP_QUERY_NOTES = 3
COMMUNITY_TOP_GOD_NODES = 8
COMMUNITY_TOP_BRIDGES = 20
FULLTEXT_INDEX_FILENAME = "fulltext_index.json"
FULLTEXT_BM25_K1 = 1.5
FULLTEXT_BM25_B = 0.75
SEMANTIC_PROMPT_VERSION = "phase2b-v1"
SEMANTIC_SECTION_MAX_CHARS = 2500
SEMANTIC_GRAPH_DISABLED_VALUES = {"0", "false", "no", "off"}
SEMANTIC_GRAPH_ENABLED_VALUES = {"1", "true", "yes", "on"}
GRAPH_STOPWORDS = {
    "this",
    "that",
    "with",
    "from",
    "have",
    "file",
    "files",
    "docs",
    "document",
    "documents",
    "guide",
    "notes",
    "readme",
    "config",
    "setting",
    "settings",
}
SUPPORTED_TEXT_SUFFIXES = {
    ".md",
    ".markdown",
    ".mdx",
    ".txt",
    ".rst",
    ".log",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".docx",
    ".xlsx",
    ".pdf",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".toml",
    ".ini",
    ".sh",
}
MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx"}
STRUCTURED_SUFFIXES = {".json", ".yaml", ".yml", ".toml", ".ini"}
PYTHON_SUFFIXES = {".py"}
_BINARY_SUFFIXES = {".docx", ".xlsx", ".pdf"}
_EXTENSION_ALIASES = {
    "python": "*.py",
    "py": "*.py",
    "yaml": "*.yaml",
    "yml": "*.yml",
    "json": "*.json",
    "toml": "*.toml",
    "ini": "*.ini",
    "md": "*.md",
    "markdown": "*.md",
    "tsx": "*.tsx",
    "ts": "*.ts",
    "jsx": "*.jsx",
    "js": "*.js",
    "java": "*.java",
    "go": "*.go",
    "rs": "*.rs",
    "shell": "*.sh",
    "sh": "*.sh",
    "docx": "*.docx",
    "xlsx": "*.xlsx",
    "xls": "*.xls",
    "pdf": "*.pdf",
}
DEFAULT_IGNORED_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    ".next",
    "target",
}
PATHISH_RE = re.compile(r"[\w./*+-]+\.[A-Za-z0-9]+|[\w./*+-]+/[\w./*+-]+")
TOKEN_RE = re.compile(r"[A-Za-z0-9_./*:-]{2,}")
NON_ASCII_FILENAME_RE = re.compile(r"[^\s`\"'“”]+[^\x00-\x7F][^\s`\"'“”]*\.[A-Za-z][A-Za-z0-9]{0,7}")
SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
WORD_CHARS_RE = re.compile(r"^[A-Za-z0-9_]+$")
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HTML_HREF_RE = re.compile(r"""href=["']([^"']+)["']""", re.I)
CODE_SPAN_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")
FULLTEXT_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]{2,}")
FULLTEXT_CJK_RUN_RE = re.compile(r"[\u3400-\u9fff]{2,}")


# ─────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────
@dataclass(slots=True)
class ChunkSpec:
    text: str
    line_start: int | None = None
    line_end: int | None = None
    label: str = ""
    section_path: tuple[str, ...] = field(default_factory=tuple)
    kind: str = ""


@dataclass(slots=True)
class IndexedFile:
    rel_path: str
    suffix: str
    kb: str
    file_path: Path
    normalized_text: str | None = None
    chunks: list[ChunkSpec] | None = None
    symbols: list[dict[str, Any]] = field(default_factory=list)
    normalized_text_path: Path | None = None
    chunk_strategy: str = ""
    _chunk_total: int = 0
    _original_chunk_total: int = 0
    _line_total: int = 0

    def get_normalized_text(self) -> str:
        if self.normalized_text is not None:
            return self.normalized_text
        if self.normalized_text_path is None:
            return ""
        return self.normalized_text_path.read_text(encoding="utf-8")

    def iter_normalized_lines(self) -> Iterable[str]:
        if self.normalized_text is not None:
            return iter(self.normalized_text.splitlines())
        if self.normalized_text_path is None:
            return iter(())
        return _iter_text_file_lines(self.normalized_text_path)

    def iter_chunks(self) -> Iterable[ChunkSpec]:
        if self.chunks is not None:
            return iter(self.chunks)
        if self.normalized_text_path is not None and self.normalized_text is None:
            return _iter_chunks_from_cached_file(
                self.normalized_text_path,
                self.suffix,
                self.chunk_strategy,
            )
        return iter(_chunks_for_indexed_text(self.get_normalized_text(), self.suffix, self.chunk_strategy))

    @property
    def chunk_count(self) -> int:
        if self._chunk_total > 0:
            return self._chunk_total
        if self.chunks is not None:
            return len(self.chunks)
        return sum(1 for _ in self.iter_chunks())

    @property
    def line_count(self) -> int:
        if self._line_total > 0:
            return self._line_total
        return max(1, sum(1 for _ in self.iter_normalized_lines()))

    @property
    def original_chunk_count(self) -> int:
        if self._original_chunk_total > 0:
            return self._original_chunk_total
        return self.chunk_count

    @property
    def truncated(self) -> bool:
        return self.original_chunk_count > self.chunk_count

    def write_normalized_text(self, target_path: Path) -> None:
        if self.normalized_text_path is not None and self.normalized_text is None:
            shutil.copyfile(self.normalized_text_path, target_path)
            return
        target_path.write_text(self.get_normalized_text(), encoding="utf-8")


@dataclass(slots=True)
class QueryPlan:
    symbols: list[str]
    keywords: list[str]
    path_globs: list[str]
    semantic_query: str
    reason: str = ""


@dataclass(slots=True)
class SearchStepResult:
    grouped_hits: dict[str, list["SearchHit"]]
    hits: list["SearchHit"]
    trace: dict[str, Any]


@dataclass(slots=True)
class GraphExpansionResult:
    sources: set[str]
    seed_sources: list[str]
    expanded_sources: list[str]
    edge_reasons: list[dict[str, Any]]
    hops: int = 0
    strategy: str = "heuristic"
    bridge_entities: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class SearchHit:
    source: str
    match_kind: str
    snippet: str
    score: float
    line_start: int | None = None
    line_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def dedupe_key(self) -> tuple[str, int | None, int | None, str]:
        normalized = " ".join(self.snippet.lower().split())
        return (self.source, self.line_start, self.line_end, normalized)


# ─────────────────────────────────────────────────────────
# Embedding / LLM 工厂
# ─────────────────────────────────────────────────────────
def _is_rate_limit_error(exc: Exception) -> bool:
    """粗略识别第三方 embedding 服务的限流错误。"""
    message = str(exc).lower()
    return "429" in message or "rate limit" in message or "tpm limit" in message


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if not value:
        return default
    if value in SEMANTIC_GRAPH_ENABLED_VALUES:
        return True
    if value in SEMANTIC_GRAPH_DISABLED_VALUES:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _configured_chunk_size() -> int:
    return max(1, _env_int("CHUNK_SIZE", GENERIC_CHUNK_SIZE))


def _configured_chunk_overlap_lines() -> int:
    return max(0, _env_int("CHUNK_OVERLAP_LINES", GENERIC_CHUNK_OVERLAP_LINES))


def _configured_max_chunks_per_file() -> int | None:
    value = _env_int("MAX_CHUNKS_PER_FILE", 0)
    if value <= 0:
        return None
    return value


def _limit_prefers_tail(chunk_strategy: str) -> bool:
    return chunk_strategy == "wechat_markdown"


def make_embeddings(
    api_key: str,
    base_url: str = SILICONFLOW_BASE_URL,
    model: str = DEFAULT_EMBED_MODEL,
) -> OpenAIEmbeddings:
    """创建向量化器（兼容任意 OpenAI 格式 Embedding 接口）"""
    return OpenAIEmbeddings(
        model=model,
        base_url=base_url,
        api_key=api_key,
        check_embedding_ctx_length=False,
    )


def make_llm(
    api_key: str,
    model: str,
    base_url: str,
    temperature: float = 0.3,
) -> ChatOpenAI:
    """创建 OpenAI 兼容的 LLM 客户端。"""
    return ChatOpenAI(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
    )


# ─────────────────────────────────────────────────────────
# 基础工具
# ─────────────────────────────────────────────────────────
def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _line_range_label(line_start: int | None, line_end: int | None) -> str:
    if line_start is None:
        return ""
    if line_end is None or line_end == line_start:
        return f"L{line_start}"
    return f"L{line_start}-L{line_end}"


def _chunk_location_label(chunk: ChunkSpec, fallback: str) -> str:
    if chunk.label:
        return chunk.label
    line_label = _line_range_label(chunk.line_start, chunk.line_end)
    return line_label or fallback


def _dedupe_strings(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []
    for item in items:
        clean = item.strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(clean)
    return results


def _normalize_source_path(path_like: str | Path) -> str:
    text = str(path_like).strip()
    if not text:
        return ""
    normalized = posixpath.normpath(text.replace("\\", "/"))
    return "" if normalized in {"", "."} else normalized


def _load_extra_exclude_globs() -> list[str]:
    raw = os.getenv("EXCLUDE_GLOBS", "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _split_lines_into_chunks(
    lines: list[str],
    start_line: int = 1,
    chunk_size: int | None = None,
    overlap_lines: int | None = None,
    label: str = "",
    section_path: tuple[str, ...] = (),
    kind: str = "",
) -> list[ChunkSpec]:
    if not lines:
        return []
    chunk_size = _configured_chunk_size() if chunk_size is None else max(1, chunk_size)
    overlap_lines = _configured_chunk_overlap_lines() if overlap_lines is None else max(0, overlap_lines)

    chunks: list[ChunkSpec] = []
    index = 0
    total = len(lines)
    while index < total:
        current_len = 0
        end = index
        while end < total:
            next_len = current_len + len(lines[end]) + 1
            if end > index and next_len > chunk_size:
                break
            current_len = next_len
            end += 1
        text = "\n".join(lines[index:end]).strip()
        if text:
            chunk_label = label or f"chunk {len(chunks) + 1}"
            chunks.append(
                ChunkSpec(
                    text=text,
                    line_start=start_line + index,
                    line_end=start_line + end - 1,
                    label=chunk_label,
                    section_path=section_path,
                    kind=kind,
                )
            )
        if end >= total:
            break
        index = max(index + 1, end - overlap_lines)
    return chunks


def _limit_chunk_specs(
    chunks: Iterable[ChunkSpec],
    max_chunks: int | None = None,
    chunk_strategy: str = "",
) -> list[ChunkSpec]:
    limit = _configured_max_chunks_per_file() if max_chunks is None else max_chunks
    items = list(chunks)
    if limit is None or len(items) <= limit:
        return items
    if _limit_prefers_tail(chunk_strategy):
        return items[-limit:]
    return items[:limit]


def _iter_limited_chunks(
    chunks: Iterable[ChunkSpec],
    max_chunks: int | None = None,
    chunk_strategy: str = "",
) -> Iterable[ChunkSpec]:
    limit = _configured_max_chunks_per_file() if max_chunks is None else max_chunks
    if limit is None:
        yield from chunks
        return
    if _limit_prefers_tail(chunk_strategy):
        tail: deque[ChunkSpec] = deque(maxlen=limit)
        for chunk in chunks:
            tail.append(chunk)
        yield from tail
        return
    emitted = 0
    for chunk in chunks:
        if emitted >= limit:
            break
        emitted += 1
        yield chunk


def _ensure_chunk_size(
    chunks: list[ChunkSpec],
    chunk_size: int | None = None,
    overlap_lines: int | None = None,
) -> list[ChunkSpec]:
    chunk_size = _configured_chunk_size() if chunk_size is None else max(1, chunk_size)
    overlap_lines = _configured_chunk_overlap_lines() if overlap_lines is None else max(0, overlap_lines)
    results: list[ChunkSpec] = []
    for chunk in chunks:
        if len(chunk.text) <= chunk_size:
            results.append(chunk)
            continue
        lines = chunk.text.splitlines()
        start_line = chunk.line_start or 1
        results.extend(
            _split_lines_into_chunks(
                lines=lines,
                start_line=start_line,
                chunk_size=chunk_size,
                overlap_lines=overlap_lines,
                label=chunk.label,
                section_path=chunk.section_path,
                kind=chunk.kind,
            )
        )
    return results


# ─────────────────────────────────────────────────────────
# 微信样式 Markdown 解析
# ─────────────────────────────────────────────────────────
_HEADER_RE = re.compile(
    r"^\*\*(.+?)\*\*\s+`\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]`"
)
_WECHAT_IMAGE_LINE_RE = re.compile(r"^!\[[^\]]*\]\([^)]+\.(?:png|jpe?g|gif|webp|bmp)\)$", re.IGNORECASE)
_WECHAT_SYSTEM_DROP_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"撤回了一条消息",
        r"邀请你加入了群聊",
        r"加入了群聊",
        r"移出了群聊",
        r"修改群名为",
        r"更换了群头像",
        r"拍了拍",
        r"开启了群待办",
        r"结束了群待办",
    )
]


def _sanitize_wechat_message_lines(lines: Iterable[str]) -> list[str]:
    sanitized: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line == "---" or line.startswith("> 导出时间"):
            continue
        if _WECHAT_IMAGE_LINE_RE.fullmatch(line):
            continue
        if line.startswith("📸 `") and "加密高级图片" in line:
            continue
        if line.startswith("📦 `") and "未知格式内建消息" in line:
            continue
        if "<![CDATA[" in line and any(
            marker in line for marker in ("<template>", "<link_list>", "<memberlist>", "<plain>")
        ):
            continue
        sanitized.append(line)
    return sanitized


def _build_wechat_message(
    sender: str | None,
    timestamp: datetime | None,
    lines: list[str],
    line_start: int | None,
    line_end: int | None,
) -> dict[str, Any] | None:
    if sender is None or line_start is None:
        return None
    cleaned_lines = _sanitize_wechat_message_lines(lines)
    if not cleaned_lines:
        return None
    body = "\n".join(cleaned_lines).strip()
    if not body:
        return None
    if sender == "系统提示" and any(pattern.search(body) for pattern in _WECHAT_SYSTEM_DROP_PATTERNS):
        return None
    return {
        "sender": sender,
        "timestamp": timestamp,
        "content": body,
        "line_start": line_start,
        "line_end": line_end or line_start,
    }


def _parse_md_messages_text(content: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    current_sender: str | None = None
    current_ts: datetime | None = None
    current_lines: list[str] = []
    current_start_line: int | None = None
    current_end_line: int | None = None

    def flush():
        message = _build_wechat_message(
            current_sender,
            current_ts,
            current_lines,
            current_start_line,
            current_end_line,
        )
        if message is not None:
            messages.append(message)

    for lineno, line in enumerate(content.splitlines(), start=1):
        match = _HEADER_RE.match(line)
        if match:
            flush()
            current_sender = match.group(1).strip()
            try:
                current_ts = datetime.strptime(match.group(2).strip(), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                current_ts = None
            current_lines = []
            current_start_line = lineno
            current_end_line = lineno
            continue

        if current_sender is None:
            continue

        stripped = line.strip()
        if stripped:
            current_lines.append(stripped)
            current_end_line = lineno

    flush()
    return messages


def _chunk_by_time_window(
    messages: list[dict[str, Any]],
    window_minutes: int = TIME_WINDOW_MINUTES,
) -> list[list[dict[str, Any]]]:
    if not messages:
        return []

    chunks: list[list[dict[str, Any]]] = []
    current_chunk: list[dict[str, Any]] = []
    window_start: datetime | None = None

    for msg in messages:
        ts = msg.get("timestamp")
        if ts is None:
            current_chunk.append(msg)
            continue

        if window_start is None:
            window_start = ts

        if ts - window_start <= timedelta(minutes=window_minutes):
            current_chunk.append(msg)
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [msg]
            window_start = ts

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


# ─────────────────────────────────────────────────────────
# 结构化文件处理
# ─────────────────────────────────────────────────────────
def _flatten_mapping(value: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten_mapping(child, child_prefix))
        return lines
    if isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            lines.extend(_flatten_mapping(child, child_prefix))
        return lines
    scalar = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    if prefix:
        lines.append(f"{prefix} = {scalar}")
    else:
        lines.append(str(scalar))
    return lines

def _extract_kb(rel_path: str) -> str:
    parts = [part for part in _normalize_source_path(rel_path).split("/") if part]
    if len(parts) > 1:
        return parts[0]
    return ""

def _normalize_graph_token(token: str) -> str | None:
    clean = token.strip().strip("`\"'").rstrip("()")
    if len(clean) < GRAPH_SHARED_TOKEN_MIN_LENGTH:
        return None
    if not WORD_CHARS_RE.fullmatch(clean):
        return None
    lowered = clean.lower()
    if lowered in GRAPH_STOPWORDS:
        return None
    return lowered


def _extract_shared_tokens(indexed: IndexedFile) -> set[str]:
    tokens: set[str] = set()

    for record in indexed.symbols:
        for raw in (record.get("name", ""), record.get("qualified_name", "")):
            for piece in re.split(r"[.\[\]]+", str(raw)):
                token = _normalize_graph_token(piece)
                if token:
                    tokens.add(token)

    for line in indexed.iter_normalized_lines():
        if indexed.suffix in STRUCTURED_SUFFIXES:
            if " = " in line:
                key = line.split(" = ", 1)[0].strip()
                if re.fullmatch(r"[A-Za-z0-9_.-]{3,}", key):
                    tokens.add(key.lower())
                for piece in re.split(r"[.\[\]-]+", key):
                    token = _normalize_graph_token(piece)
                    if token:
                        tokens.add(token)

        for match in CODE_SPAN_RE.findall(line):
            token = _normalize_graph_token(match)
            if token:
                tokens.add(token)

        for match in CALL_RE.findall(line):
            token = _normalize_graph_token(match)
            if token:
                tokens.add(token)

    return tokens


def _iter_local_path_references(text: str) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    refs: list[tuple[str, str]] = []
    for ref in MARKDOWN_LINK_RE.findall(text):
        item = ("links_to", ref)
        if item not in seen:
            seen.add(item)
            refs.append(item)
    for ref in HTML_HREF_RE.findall(text):
        item = ("links_to", ref)
        if item not in seen:
            seen.add(item)
            refs.append(item)
    for ref in PATHISH_RE.findall(text):
        item = ("mentions_path", ref)
        if item not in seen:
            seen.add(item)
            refs.append(item)
    return refs


def _iter_local_path_references_in_lines(lines: Iterable[str]) -> Iterable[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    for line in lines:
        for ref in MARKDOWN_LINK_RE.findall(line):
            item = ("links_to", ref)
            if item in seen:
                continue
            seen.add(item)
            yield item
        for ref in HTML_HREF_RE.findall(line):
            item = ("links_to", ref)
            if item in seen:
                continue
            seen.add(item)
            yield item
        for ref in PATHISH_RE.findall(line):
            item = ("mentions_path", ref)
            if item in seen:
                continue
            seen.add(item)
            yield item


def _resolve_document_reference(
    source: str,
    reference: str,
    source_lookup: set[str],
    basename_lookup: dict[str, list[str]],
) -> str | None:
    clean = reference.strip().strip("<>").strip("'\"")
    if not clean or clean.startswith("#"):
        return None
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", clean) or clean.startswith("mailto:"):
        return None

    clean = _normalize_source_path(clean.split("#", 1)[0].split("?", 1)[0])
    if not clean:
        return None
    if clean in source_lookup:
        return clean

    source = _normalize_source_path(source)
    if clean.startswith("/"):
        candidate = posixpath.normpath(clean.lstrip("/"))
    else:
        parent = posixpath.dirname(source)
        candidate = posixpath.normpath(posixpath.join(parent, clean))
    if candidate in source_lookup:
        return candidate

    basename = posixpath.basename(clean)
    matches = basename_lookup.get(basename, [])
    if len(matches) == 1:
        return matches[0]
    return None


def _stem_tokens(stem: str) -> set[str]:
    return {
        token
        for token in re.split(r"[_\-.]", stem.lower())
        if len(token) >= GRAPH_SHARED_TOKEN_MIN_LENGTH
    }


def _add_graph_edge(
    edges_by_source: dict[str, dict[str, tuple[int, dict[str, Any]]]],
    source: str,
    target: str,
    kind: str,
    reason: str,
) -> None:
    if not source or not target or source == target:
        return
    priority = GRAPH_EDGE_PRIORITY[kind]
    payload = {"target": target, "kind": kind, "reason": reason}
    current = edges_by_source[source].get(target)
    if current is None or priority < current[0]:
        edges_by_source[source][target] = (priority, payload)

def _entity_file_node_id(source: str) -> str:
    return f"file:{_normalize_source_path(source)}"


def _entity_section_node_id(source: str, chunk_index: int) -> str:
    return f"section:{_normalize_source_path(source)}:{chunk_index}"


def _entity_symbol_node_id(record: dict[str, Any]) -> str:
    source = _normalize_source_path(record.get("source", ""))
    qualified_name = str(record.get("qualified_name") or record.get("name") or "symbol")
    symbol_kind = str(record.get("kind") or "symbol")
    line_start = int(record.get("line_start") or 0)
    return f"symbol:{source}:{symbol_kind}:{qualified_name}:{line_start}"


def _entity_query_note_node_id(note_relpath: str) -> str:
    return f"query_note:{_normalize_source_path(note_relpath)}"


def _entity_edge_confidence(kind: str) -> str:
    return "INFERRED" if kind in ENTITY_INFERRED_EDGE_TYPES else "EXTRACTED"


def _add_entity_node(
    nodes: list[dict[str, Any]],
    seen: set[str],
    payload: dict[str, Any],
) -> None:
    node_id = str(payload.get("id", "")).strip()
    if not node_id or node_id in seen:
        return
    seen.add(node_id)
    nodes.append(payload)


def _add_entity_edge(
    edges: dict[tuple[Any, ...], dict[str, Any]],
    source_id: str,
    target_id: str,
    kind: str,
    evidence_source: str,
    reason: str = "",
    line_start: int | None = None,
    line_end: int | None = None,
) -> None:
    source_id = str(source_id).strip()
    target_id = str(target_id).strip()
    evidence_source = _normalize_source_path(evidence_source)
    if not source_id or not target_id or source_id == target_id or not evidence_source:
        return

    key = (source_id, target_id, kind, reason, evidence_source, line_start, line_end)
    if key in edges:
        return

    evidence: dict[str, Any] = {"source": evidence_source}
    if line_start is not None:
        evidence["line_start"] = line_start
    if line_end is not None:
        evidence["line_end"] = line_end

    payload: dict[str, Any] = {
        "source": source_id,
        "target": target_id,
        "type": kind,
        "confidence": _entity_edge_confidence(kind),
        "evidence": evidence,
    }
    if reason:
        payload["reason"] = reason
    edges[key] = payload


def _entity_bridge_payload(node: dict[str, Any], relation: str = "") -> dict[str, Any]:
    payload = {
        "id": str(node.get("id", "")),
        "type": str(node.get("type", "")),
        "name": str(node.get("qualified_name") or node.get("name") or ""),
        "source": str(node.get("source", "") or ""),
    }
    if node.get("line_start") is not None:
        payload["line_start"] = node.get("line_start")
    if node.get("line_end") is not None:
        payload["line_end"] = node.get("line_end")
    if relation:
        payload["relation"] = relation
    return payload


def _add_bridge_entity(
    bridge_entities: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    payload: dict[str, Any],
) -> None:
    entity_id = str(payload.get("id", ""))
    relation = str(payload.get("relation", ""))
    key = (entity_id, relation)
    if not entity_id or key in seen:
        return
    seen.add(key)
    bridge_entities.append(payload)


def _entity_expansion_priority(kind: str) -> int:
    return ENTITY_EXPANSION_PRIORITY.get(kind, len(ENTITY_EXPANSION_PRIORITY) + 10)


def _entity_node_file_source(node: dict[str, Any]) -> str:
    for key in ("file", "source", "path"):
        value = node.get(key)
        if isinstance(value, str):
            clean = _normalize_source_path(value)
            if clean:
                return clean
    return ""


def _entity_file_neighbors(
    bundle: SearchBundle,
    source: str,
    valid_sources: set[str] | None = None,
) -> list[dict[str, Any]]:
    source = _normalize_source_path(source)
    file_node_id = _entity_file_node_id(source)
    if not source or file_node_id not in bundle.entity_nodes_by_id:
        return []

    source_kb = _extract_kb(source)
    best_by_target: dict[str, tuple[tuple[int, int, str], dict[str, Any]]] = {}

    def consider(
        target_source: str,
        kind: str,
        reason: str,
        bridges: list[dict[str, Any]],
    ) -> None:
        target_source = _normalize_source_path(target_source)
        if not target_source or target_source == source:
            return
        if valid_sources is not None and target_source not in valid_sources:
            return
        priority = (
            _entity_expansion_priority(kind),
            0 if _extract_kb(target_source) == source_kb else 1,
            target_source,
        )
        payload = {
            "to": target_source,
            "kind": kind,
            "reason": reason,
            "bridges": bridges,
        }
        current = best_by_target.get(target_source)
        if current is None or priority < current[0]:
            best_by_target[target_source] = (priority, payload)

    def target_node_for(edge: dict[str, Any]) -> dict[str, Any] | None:
        target_id = str(edge.get("target", "") or "")
        return bundle.entity_nodes_by_id.get(target_id)

    def consider_semantic_neighbors(
        semantic_node: dict[str, Any],
        bridges: list[dict[str, Any]],
        visited_semantic: set[str],
        anchor_source: str,
    ) -> None:
        semantic_node_id = str(semantic_node.get("id", "") or "")
        if not semantic_node_id or semantic_node_id in visited_semantic:
            return
        next_visited = set(visited_semantic)
        next_visited.add(semantic_node_id)
        for semantic_edge in bundle.entity_edges_by_source.get(semantic_node_id, []):
            semantic_target = target_node_for(semantic_edge)
            if not semantic_target:
                continue
            semantic_kind = str(semantic_edge.get("type", "") or "")
            semantic_reason = str(semantic_edge.get("reason", "") or "")
            semantic_target_type = str(semantic_target.get("type", "") or "")
            evidence_source = _entity_edge_evidence_source(semantic_edge)
            if (
                str(semantic_node.get("type", "") or "") == "query_note"
                and semantic_target_type in {"concept", "decision", "query_note"}
                and evidence_source
                and evidence_source != anchor_source
            ):
                continue
            if semantic_target_type == "file":
                consider(
                    target_source=str(semantic_target.get("source", "") or ""),
                    kind=semantic_kind,
                    reason=semantic_reason,
                    bridges=bridges,
                )
            elif semantic_target_type == "section":
                section_bridge = _entity_bridge_payload(semantic_target, relation=semantic_kind)
                consider(
                    target_source=str(semantic_target.get("source", "") or ""),
                    kind=semantic_kind,
                    reason=semantic_reason,
                    bridges=[*bridges, section_bridge],
                )
            elif semantic_target_type == "symbol":
                symbol_bridge = _entity_bridge_payload(semantic_target, relation=semantic_kind)
                consider(
                    target_source=str(semantic_target.get("source", "") or ""),
                    kind=semantic_kind,
                    reason=semantic_reason,
                    bridges=[*bridges, symbol_bridge],
                )
            elif semantic_target_type in {"concept", "decision", "query_note"}:
                nested_bridge = _entity_bridge_payload(
                    semantic_target,
                    relation=semantic_kind or semantic_target_type,
                )
                consider_semantic_neighbors(
                    semantic_target,
                    [*bridges, nested_bridge],
                    next_visited,
                    anchor_source,
                )

    for edge in bundle.entity_edges_by_source.get(file_node_id, []):
        target_node = target_node_for(edge)
        if not target_node:
            continue
        edge_kind = str(edge.get("type", "") or "")
        edge_reason = str(edge.get("reason", "") or "")
        target_type = str(target_node.get("type", "") or "")

        if target_type == "file":
            consider(
                target_source=str(target_node.get("source", "") or ""),
                kind=edge_kind,
                reason=edge_reason,
                bridges=[],
            )
            continue

        if target_type == "section":
            section_bridge = _entity_bridge_payload(target_node, relation=edge_kind or "contains")
            for section_edge in bundle.entity_edges_by_source.get(str(target_node["id"]), []):
                section_target = target_node_for(section_edge)
                if not section_target:
                    continue
                section_kind = str(section_edge.get("type", "") or "")
                section_reason = str(section_edge.get("reason", "") or "")
                section_target_type = str(section_target.get("type", "") or "")
                if section_target_type == "file":
                    consider(
                        target_source=str(section_target.get("source", "") or ""),
                        kind=section_kind,
                        reason=section_reason,
                        bridges=[section_bridge],
                    )
                elif section_target_type == "symbol":
                    symbol_bridge = _entity_bridge_payload(section_target, relation=section_kind)
                    consider(
                        target_source=str(section_target.get("source", "") or ""),
                        kind=section_kind,
                        reason=section_reason,
                        bridges=[section_bridge, symbol_bridge],
                    )
            continue

        if target_type == "symbol":
            symbol_bridge = _entity_bridge_payload(target_node, relation=edge_kind or "defines")
            for symbol_edge in bundle.entity_edges_by_source.get(str(target_node["id"]), []):
                symbol_target = target_node_for(symbol_edge)
                if not symbol_target or str(symbol_target.get("type", "") or "") != "file":
                    continue
                consider(
                    target_source=str(symbol_target.get("source", "") or ""),
                    kind=str(symbol_edge.get("type", "") or ""),
                    reason=str(symbol_edge.get("reason", "") or ""),
                    bridges=[symbol_bridge],
                )
            continue

        if target_type in {"concept", "decision", "query_note"}:
            semantic_bridge = _entity_bridge_payload(target_node, relation=edge_kind or target_type)
            consider_semantic_neighbors(target_node, [semantic_bridge], set(), source)

    return [
        payload
        for _, payload in sorted(
            best_by_target.values(),
            key=lambda item: item[0],
        )
    ]


def list_kbs(persist_dir: str = DEFAULT_FAISS_DIR) -> list[str]:
    manifest_path = Path(persist_dir) / "index_manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("kb_enabled"):
        return []
    kbs: set[str] = set()
    for file_entry in manifest.get("files", []):
        kb = file_entry.get("kb") or _extract_kb(file_entry.get("name", ""))
        if kb:
            kbs.add(kb)
    return sorted(kbs)


# ─────────────────────────────────────────────────────────
# 问答
# ─────────────────────────────────────────────────────────
_PROMPT_TEMPLATE = """你是一个私人知识助手。以下参考资料供你回答时使用：

{context}

---
要求：
1. 直接给出答案，像一个领域专家一样回答
2. 不要提及“片段X”“来源文件”等内部检索过程
3. 如果信息不足以回答，直接说不确定，不要虚构
4. 用中文回答，语言简洁专业

用户问题：{question}"""


def _chunk_to_text(chunk: Any) -> str:
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)
