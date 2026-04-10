#!/usr/bin/env python3
"""
ragbot.py — 通用本地文档 RAG / Hybrid Search 核心模块

架构：
  OpenAI 兼容 Embedding API → FAISS (向量存储)
  + 本地 text cache / symbol index / hybrid retrieval
  → 任意 OpenAI 兼容 LLM (生成)
"""
from __future__ import annotations

import ast
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import configparser
import fnmatch
import hashlib
import json
import os
import posixpath
import re
from queue import Empty, Queue
import shutil
import subprocess
import tempfile
from threading import local
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import unquote

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    yaml = None
    _YAML_AVAILABLE = False

# 条件导入 markitdown
try:
    from markitdown import MarkItDown

    _MARKITDOWN_AVAILABLE = True
except ImportError:
    _MARKITDOWN_AVAILABLE = False

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


# ─────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────
@dataclass(slots=True)
class ChunkSpec:
    text: str
    line_start: int | None = None
    line_end: int | None = None
    label: str = ""


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


@dataclass(slots=True)
class SearchBundle:
    vectorstore: FAISS
    persist_dir: Path
    source_dir: Path | None
    manifest: dict[str, Any]
    files: list[dict[str, Any]]
    files_by_source: dict[str, dict[str, Any]]
    normalized_text_dir: Path
    symbol_index: list[dict[str, Any]]
    document_graph: dict[str, Any]
    graph_neighbors: dict[str, list[dict[str, Any]]]
    entity_graph: dict[str, Any]
    entity_nodes_by_id: dict[str, dict[str, Any]]
    entity_edges_by_source: dict[str, list[dict[str, Any]]]
    wiki_pages: list[dict[str, Any]]

    def cache_path_for(self, source: str) -> Path | None:
        entry = self.files_by_source.get(source)
        if not entry:
            return None
        relative = entry.get("normalized_text")
        if not relative:
            return None
        return self.normalized_text_dir / relative


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


def _runtime_wiki_artifact_paths(persist_path: Path) -> list[str]:
    wiki_dir = persist_path / WIKI_DIRNAME
    if not wiki_dir.exists():
        return []

    paths: list[str] = []
    for page_path in sorted(wiki_dir.rglob("*.md")):
        if not page_path.is_file():
            continue
        relative = page_path.relative_to(persist_path).as_posix()
        if relative == f"{WIKI_DIRNAME}/log.md":
            continue
        paths.append(relative)
    return paths


def search_bundle_artifact_signature(
    persist_dir: str | Path,
) -> tuple[tuple[str, bool, int, int], ...]:
    persist_path = Path(persist_dir)
    manifest_path = persist_path / "index_manifest.json"
    manifest: dict[str, Any] = {}
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
        str(manifest.get("symbol_index_file", SYMBOL_INDEX_FILENAME) or SYMBOL_INDEX_FILENAME),
        str(manifest.get("document_graph_file", DOCUMENT_GRAPH_FILENAME) or DOCUMENT_GRAPH_FILENAME),
        str(manifest.get("entity_graph_file", ENTITY_GRAPH_FILENAME) or ENTITY_GRAPH_FILENAME),
        str(manifest.get("community_index_file", COMMUNITY_INDEX_FILENAME) or COMMUNITY_INDEX_FILENAME),
        str(manifest.get("lint_report_file", LINT_REPORT_FILENAME) or LINT_REPORT_FILENAME),
        str(
            manifest.get("graph_report_file", f"{REPORTS_DIRNAME}/{GRAPH_REPORT_FILENAME}")
            or f"{REPORTS_DIRNAME}/{GRAPH_REPORT_FILENAME}"
        ),
    ]
    relative_paths.extend(_runtime_wiki_artifact_paths(persist_path))

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


def _markdown_heading_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return fallback


def _source_from_wiki_file_relpath(relpath: str) -> str:
    normalized = _normalize_source_path(relpath)
    if not normalized.startswith("files/") or not normalized.endswith(".md"):
        return ""
    return _normalize_source_path(normalized[len("files/") : -3])


def _wiki_page_kind(relpath: str) -> str:
    normalized = _normalize_source_path(relpath)
    if normalized == "index.md":
        return "index"
    if normalized == "log.md":
        return "log"
    if normalized.startswith("files/"):
        return "file"
    if normalized.startswith(f"{QUERY_NOTES_DIRNAME}/"):
        return "query"
    if normalized.startswith("communities/"):
        return "community"
    if normalized.startswith("entities/"):
        return "entity"
    return ""


def _source_from_wiki_link_target(target: str) -> str:
    cleaned = unquote(str(target).strip()).split("#", 1)[0].split("?", 1)[0]
    if cleaned.startswith("../"):
        cleaned = cleaned[3:]
    return _source_from_wiki_file_relpath(cleaned)


def _iter_wiki_markdown_links(text: str) -> list[str]:
    links: list[str] = []
    in_fence = False
    fence_marker = ""
    fence_len = 0
    for line in text.splitlines():
        stripped = line.lstrip()
        fence_match = re.match(r"^([`~]{3,})", stripped)
        if fence_match:
            marker_run = fence_match.group(1)
            marker = marker_run[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
                fence_len = len(marker_run)
                continue
            if marker == fence_marker and len(marker_run) >= fence_len:
                in_fence = False
                fence_marker = ""
                fence_len = 0
                continue
        if in_fence:
            continue
        links.extend(
            target.strip()
            for target in MARKDOWN_LINK_RE.findall(line)
            if isinstance(target, str) and target.strip()
        )
    return links


def _load_wiki_pages(persist_dir: Path, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wiki_dir = persist_dir / WIKI_DIRNAME
    if not wiki_dir.exists():
        return []

    file_relpaths = {
        _normalize_source_path((Path("files") / Path(f"{str(entry.get('name', '') or '')}.md")).as_posix()): _normalize_source_path(
            str(entry.get("name", "") or "")
        )
        for entry in files
        if _normalize_source_path(str(entry.get("name", "") or ""))
    }
    from wiki import is_wiki_write_in_progress

    last_error: OSError | None = None
    for attempt in range(3):
        pages: list[dict[str, Any]] = []
        try:
            for page_path in sorted(wiki_dir.rglob("*.md")):
                relpath = _normalize_source_path(page_path.relative_to(wiki_dir).as_posix())
                kind = _wiki_page_kind(relpath)
                if not kind or kind == "log":
                    continue
                text = page_path.read_text(encoding="utf-8")
                if kind == "file":
                    source = file_relpaths.get(relpath) or _source_from_wiki_file_relpath(relpath)
                    source_refs = [source] if source else []
                    title = source or _markdown_heading_title(text, page_path.stem)
                    page_id = f"wiki:file:{source or relpath}"
                else:
                    source_refs = _dedupe_strings(
                        _source_from_wiki_link_target(target)
                        for target in _iter_wiki_markdown_links(text)
                    )
                    title = "知识导航" if kind == "index" else _markdown_heading_title(text, page_path.stem)
                    page_id = f"wiki:{kind}:{relpath}"
                pages.append(
                    {
                        "id": page_id,
                        "kind": kind,
                        "title": title,
                        "relpath": relpath,
                        "text": text,
                        "source_refs": source_refs,
                    }
                )
            return pages
        except OSError as exc:
            last_error = exc
            if is_wiki_write_in_progress(persist_dir) and attempt < 2:
                time.sleep(0.05)
                continue
            raise

    if last_error is not None:
        raise last_error
    return []


def _safe_signature_from_args(node: ast.AST) -> str:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    args = []
    for arg in node.args.posonlyargs:
        args.append(arg.arg)
    for arg in node.args.args:
        args.append(arg.arg)
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    for arg in node.args.kwonlyargs:
        args.append(arg.arg)
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    return ", ".join(args)


def _load_extra_exclude_globs() -> list[str]:
    raw = os.getenv("EXCLUDE_GLOBS", "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _should_ignore_relative_path(rel_path: Path, extra_patterns: list[str]) -> bool:
    parts = set(rel_path.parts)
    if parts & DEFAULT_IGNORED_DIRS:
        return True
    posix_path = rel_path.as_posix()
    for pattern in extra_patterns:
        if fnmatch.fnmatch(posix_path, pattern) or fnmatch.fnmatch(rel_path.name, pattern):
            return True
    return False


def _iter_supported_files(source_dir: Path) -> list[Path]:
    """递归列出 source_dir 下支持的文本文件。"""
    files: list[Path] = []
    extra_patterns = _load_extra_exclude_globs()

    for dirpath, dirnames, filenames in os.walk(source_dir):
        current_dir = Path(dirpath)
        rel_dir = current_dir.relative_to(source_dir)
        kept_dirs: list[str] = []
        for dirname in dirnames:
            rel_child = rel_dir / dirname
            if _should_ignore_relative_path(rel_child, extra_patterns):
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in filenames:
            path = current_dir / filename
            rel_path = path.relative_to(source_dir)
            if _should_ignore_relative_path(rel_path, extra_patterns):
                continue
            if path.suffix.lower() in SUPPORTED_TEXT_SUFFIXES:
                files.append(path)

    return sorted(files)


def _split_lines_into_chunks(
    lines: list[str],
    start_line: int = 1,
    chunk_size: int | None = None,
    overlap_lines: int | None = None,
    label: str = "",
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


def _normalize_structured_text(text: str, suffix: str) -> str:
    try:
        if suffix == ".json":
            parsed = json.loads(text)
            return "\n".join(_flatten_mapping(parsed))
        if suffix in {".yaml", ".yml"} and _YAML_AVAILABLE:
            parsed = yaml.safe_load(text)
            return "\n".join(_flatten_mapping(parsed))
        if suffix == ".toml":
            parsed = tomllib.loads(text)
            return "\n".join(_flatten_mapping(parsed))
        if suffix == ".ini":
            parser = configparser.ConfigParser()
            parser.optionxform = str
            parser.read_string(text)
            data: dict[str, Any] = {}
            for section in parser.sections():
                data[section] = dict(parser.items(section))
            return "\n".join(_flatten_mapping(data))
    except Exception:
        pass
    return _normalize_text(text)


def _chunk_structured_text(normalized_text: str) -> list[ChunkSpec]:
    return _split_lines_into_chunks(normalized_text.splitlines())


def _convert_binary_to_markdown(file_path: Path) -> str:
    if not _MARKITDOWN_AVAILABLE:
        raise ImportError(
            f"处理 {file_path.suffix} 文件需要 markitdown 库，"
            "请运行: pip install 'markitdown[pdf,docx,xlsx]'"
        )
    md = MarkItDown()
    result = md.convert(str(file_path))
    return result.text_content


def _chunk_markdown_by_heading(text: str) -> list[ChunkSpec]:
    lines = text.splitlines()
    if not lines:
        return []

    heading_indexes = [
        idx for idx, line in enumerate(lines) if re.match(r"^\s{0,3}#{1,6}\s+", line)
    ]
    if not heading_indexes:
        return _split_lines_into_chunks(lines)

    chunks: list[ChunkSpec] = []
    starts = heading_indexes + [len(lines)]
    for current, next_index in zip(starts, starts[1:]):
        section_lines = lines[current:next_index]
        section_text = "\n".join(section_lines).strip()
        if not section_text:
            continue
        heading_line = section_lines[0].strip()
        label = heading_line.lstrip("#").strip() or f"section {len(chunks) + 1}"
        chunks.append(
            ChunkSpec(
                text=section_text,
                line_start=current + 1,
                line_end=next_index,
                label=label,
            )
        )

    preface_end = heading_indexes[0]
    if preface_end > 0:
        preface_lines = lines[:preface_end]
        preface_text = "\n".join(preface_lines).strip()
        if preface_text:
            chunks.insert(
                0,
                ChunkSpec(
                    text=preface_text,
                    line_start=1,
                    line_end=preface_end,
                    label="preface",
                ),
            )

    return _ensure_chunk_size(chunks)


def _iter_text_file_lines(path: Path) -> Iterable[str]:
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.endswith("\r"):
                line = line[:-1]
            yield line


def _emit_line_chunk(
    lines: list[str],
    start_line: int | None,
    label: str = "",
) -> ChunkSpec | None:
    if not lines or start_line is None:
        return None
    text = "\n".join(lines).strip()
    if not text:
        return None
    return ChunkSpec(
        text=text,
        line_start=start_line,
        line_end=start_line + len(lines) - 1,
        label=label,
    )


def _iter_chunk_specs_from_line_stream(
    numbered_lines: Iterable[tuple[int, str]],
    chunk_size: int | None = None,
    overlap_lines: int | None = None,
    label: str = "",
) -> Iterable[ChunkSpec]:
    chunk_size = _configured_chunk_size() if chunk_size is None else max(1, chunk_size)
    overlap_lines = _configured_chunk_overlap_lines() if overlap_lines is None else max(0, overlap_lines)
    window: list[str] = []
    window_start: int | None = None
    current_len = 0

    for line_no, line in numbered_lines:
        line_len = len(line) + 1
        if window and current_len + line_len > chunk_size:
            chunk = _emit_line_chunk(window, window_start, label=label)
            if chunk is not None:
                yield chunk
            if overlap_lines > 0:
                window = window[-overlap_lines:]
                current_len = sum(len(item) + 1 for item in window)
                window_start = line_no - len(window)
            else:
                window = []
                current_len = 0
                window_start = None
        if not window:
            window_start = line_no
        window.append(line)
        current_len += line_len

    chunk = _emit_line_chunk(window, window_start, label=label)
    if chunk is not None:
        yield chunk

def _iter_markdown_heading_chunks_from_file(path: Path) -> Iterable[ChunkSpec]:
    heading_re = re.compile(r"^\s{0,3}#{1,6}\s+")
    chunk_size = _configured_chunk_size()
    overlap_lines = _configured_chunk_overlap_lines()
    segment_lines: list[str] = []
    segment_start_line: int | None = None
    segment_label = "preface"
    current_len = 0
    saw_heading = False

    def push_line(line_no: int, line: str) -> Iterable[ChunkSpec]:
        nonlocal segment_lines, segment_start_line, current_len
        line_len = len(line) + 1
        if segment_lines and current_len + line_len > chunk_size:
            chunk = _emit_line_chunk(segment_lines, segment_start_line, label=segment_label)
            if chunk is not None:
                yield chunk
            if overlap_lines > 0:
                segment_lines = segment_lines[-overlap_lines:]
                current_len = sum(len(item) + 1 for item in segment_lines)
                segment_start_line = line_no - len(segment_lines)
            else:
                segment_lines = []
                current_len = 0
                segment_start_line = None
        if not segment_lines:
            segment_start_line = line_no
        segment_lines.append(line)
        current_len += line_len

    def flush_segment() -> Iterable[ChunkSpec]:
        nonlocal segment_lines, segment_start_line, current_len
        chunk = _emit_line_chunk(segment_lines, segment_start_line, label=segment_label)
        if chunk is not None:
            yield chunk
        segment_lines = []
        segment_start_line = None
        current_len = 0

    for line_no, line in enumerate(_iter_text_file_lines(path), start=1):
        if heading_re.match(line):
            if segment_lines:
                yield from flush_segment()
            saw_heading = True
            segment_label = line.strip().lstrip("#").strip() or "section"
            yield from push_line(line_no, line)
            continue

        if not saw_heading:
            segment_label = "preface"
        yield from push_line(line_no, line)

    if segment_lines:
        yield from flush_segment()


def _iter_wechat_markdown_chunks_from_file(
    path: Path,
    window_minutes: int = TIME_WINDOW_MINUTES,
) -> Iterable[ChunkSpec]:
    current_sender: str | None = None
    current_ts: datetime | None = None
    current_lines: list[str] = []
    current_start_line: int | None = None
    current_end_line: int | None = None
    current_chunk: list[dict[str, Any]] = []
    window_start: datetime | None = None
    chunk_index = 0

    def flush_message() -> dict[str, Any] | None:
        return _build_wechat_message(
            current_sender,
            current_ts,
            current_lines,
            current_start_line,
            current_end_line,
        )

    def emit_chunk(messages: list[dict[str, Any]]) -> ChunkSpec | None:
        nonlocal chunk_index
        if not messages:
            return None
        chunk_index += 1
        chunk_lines = []
        for message in messages:
            ts = message["timestamp"].strftime("%H:%M") if message.get("timestamp") else "?"
            chunk_lines.append(f"[{ts}] {message['sender']}: {message['content']}")
        text = "\n".join(chunk_lines).strip()
        if not text:
            return None
        timestamps = [item["timestamp"] for item in messages if item.get("timestamp")]
        start_t = timestamps[0].strftime("%Y-%m-%d %H:%M") if timestamps else ""
        end_t = timestamps[-1].strftime("%H:%M") if timestamps else ""
        label = f"{start_t} ~ {end_t}" if start_t else f"window {chunk_index}"
        return ChunkSpec(
            text=text,
            line_start=messages[0].get("line_start"),
            line_end=messages[-1].get("line_end"),
            label=label,
        )

    for lineno, line in enumerate(_iter_text_file_lines(path), start=1):
        match = _HEADER_RE.match(line)
        if match:
            message = flush_message()
            if message is not None:
                ts = message.get("timestamp")
                if ts is None:
                    current_chunk.append(message)
                else:
                    if window_start is None:
                        window_start = ts
                    if ts - window_start <= timedelta(minutes=window_minutes):
                        current_chunk.append(message)
                    else:
                        chunk = emit_chunk(current_chunk)
                        if chunk is not None:
                            yield from _ensure_chunk_size([chunk])
                        current_chunk = [message]
                        window_start = ts
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

    message = flush_message()
    if message is not None:
        ts = message.get("timestamp")
        if ts is None:
            current_chunk.append(message)
        else:
            if window_start is None or ts - window_start <= timedelta(minutes=window_minutes):
                current_chunk.append(message)
            else:
                chunk = emit_chunk(current_chunk)
                if chunk is not None:
                    yield from _ensure_chunk_size([chunk])
                current_chunk = [message]
                window_start = ts

    chunk = emit_chunk(current_chunk)
    if chunk is not None:
        yield from _ensure_chunk_size([chunk])


def _iter_chunks_from_cached_file(
    path: Path,
    suffix: str,
    chunk_strategy: str,
) -> Iterable[ChunkSpec]:
    if chunk_strategy == "wechat_markdown":
        return _iter_limited_chunks(_iter_wechat_markdown_chunks_from_file(path), chunk_strategy=chunk_strategy)
    if chunk_strategy == "markdown_heading":
        return _iter_limited_chunks(_iter_markdown_heading_chunks_from_file(path), chunk_strategy=chunk_strategy)
    if chunk_strategy in {"structured", "generic"}:
        return _iter_limited_chunks(
            _iter_chunk_specs_from_line_stream(enumerate(_iter_text_file_lines(path), start=1)),
            chunk_strategy=chunk_strategy,
        )
    if chunk_strategy == "python":
        return iter(_limit_chunk_specs(_chunk_python_code(path.read_text(encoding="utf-8")), chunk_strategy=chunk_strategy))
    return iter(_chunks_for_indexed_text(path.read_text(encoding="utf-8"), suffix, chunk_strategy))


def _chunk_python_code(text: str) -> list[ChunkSpec]:
    normalized = _normalize_text(text)
    try:
        tree = ast.parse(normalized)
    except SyntaxError:
        return _split_lines_into_chunks(normalized.splitlines())

    lines = normalized.splitlines()
    body_nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    if not body_nodes:
        return _split_lines_into_chunks(lines)

    chunks: list[ChunkSpec] = []
    preface_end = body_nodes[0].lineno - 1
    if preface_end > 0:
        chunks.extend(
            _split_lines_into_chunks(lines[:preface_end], start_line=1, label="module prelude")
        )

    for node in body_nodes:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", start)
        if start is None:
            continue
        node_lines = lines[start - 1 : end]
        label = getattr(node, "name", f"block {len(chunks) + 1}")
        if isinstance(node, ast.ClassDef):
            label = f"class {label}"
        else:
            label = f"def {label}"
        chunks.append(
            ChunkSpec(
                text="\n".join(node_lines).strip(),
                line_start=start,
                line_end=end,
                label=label,
            )
        )

    return _ensure_chunk_size(chunks)


def _extract_python_symbols(text: str, rel_path: str) -> list[dict[str, Any]]:
    normalized = _normalize_text(text)
    try:
        tree = ast.parse(normalized)
    except SyntaxError:
        return []

    records: list[dict[str, Any]] = []
    total_lines = max(1, len(normalized.splitlines()))
    records.append(
        {
            "kind": "module",
            "name": Path(rel_path).stem,
            "qualified_name": Path(rel_path).stem,
            "source": rel_path,
            "line_start": 1,
            "line_end": total_lines,
            "signature": f"module {rel_path}",
        }
    )

    class Collector(ast.NodeVisitor):
        def __init__(self):
            self.stack: list[str] = []

        def _qualified_name(self, name: str) -> str:
            return ".".join([*self.stack, name]) if self.stack else name

        def visit_ClassDef(self, node: ast.ClassDef) -> Any:
            qualified = self._qualified_name(node.name)
            records.append(
                {
                    "kind": "class",
                    "name": node.name,
                    "qualified_name": qualified,
                    "source": rel_path,
                    "line_start": node.lineno,
                    "line_end": getattr(node, "end_lineno", node.lineno),
                    "signature": f"class {node.name}",
                }
            )
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            qualified = self._qualified_name(node.name)
            signature = _safe_signature_from_args(node)
            records.append(
                {
                    "kind": "function",
                    "name": node.name,
                    "qualified_name": qualified,
                    "source": rel_path,
                    "line_start": node.lineno,
                    "line_end": getattr(node, "end_lineno", node.lineno),
                    "signature": f"def {qualified}({signature})",
                }
            )
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
            qualified = self._qualified_name(node.name)
            signature = _safe_signature_from_args(node)
            records.append(
                {
                    "kind": "function",
                    "name": node.name,
                    "qualified_name": qualified,
                    "source": rel_path,
                    "line_start": node.lineno,
                    "line_end": getattr(node, "end_lineno", node.lineno),
                    "signature": f"async def {qualified}({signature})",
                }
            )
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Import(self, node: ast.Import) -> Any:
            for alias in node.names:
                records.append(
                    {
                        "kind": "import",
                        "name": alias.asname or alias.name,
                        "qualified_name": alias.name,
                        "source": rel_path,
                        "line_start": node.lineno,
                        "line_end": getattr(node, "end_lineno", node.lineno),
                        "signature": f"import {alias.name}",
                    }
                )

        def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
            module = node.module or ""
            for alias in node.names:
                name = alias.asname or alias.name
                qualified = f"{module}.{alias.name}".strip(".")
                records.append(
                    {
                        "kind": "import",
                        "name": name,
                        "qualified_name": qualified,
                        "source": rel_path,
                        "line_start": node.lineno,
                        "line_end": getattr(node, "end_lineno", node.lineno),
                        "signature": f"from {module} import {alias.name}",
                    }
                )

    Collector().visit(tree)
    return records


def _extract_kb(rel_path: str) -> str:
    parts = [part for part in _normalize_source_path(rel_path).split("/") if part]
    if len(parts) > 1:
        return parts[0]
    return ""


def _prepare_indexed_content(
    rel_path: str,
    suffix: str,
    normalized_raw: str,
) -> tuple[str, list[ChunkSpec], list[dict[str, Any]], str, int, int]:
    symbols: list[dict[str, Any]] = []
    chunk_strategy = "generic"
    if suffix in MARKDOWN_SUFFIXES or suffix in _BINARY_SUFFIXES:
        messages = _parse_md_messages_text(normalized_raw) if suffix in MARKDOWN_SUFFIXES else []
        if messages:
            chunk_strategy = "wechat_markdown"
            chunk_specs: list[ChunkSpec] = []
            for idx, chunk in enumerate(_chunk_by_time_window(messages), start=1):
                chunk_lines = []
                for message in chunk:
                    ts = message["timestamp"].strftime("%H:%M") if message.get("timestamp") else "?"
                    chunk_lines.append(f"[{ts}] {message['sender']}: {message['content']}")
                text = "\n".join(chunk_lines).strip()
                if not text:
                    continue
                timestamps = [item["timestamp"] for item in chunk if item.get("timestamp")]
                start_t = timestamps[0].strftime("%Y-%m-%d %H:%M") if timestamps else ""
                end_t = timestamps[-1].strftime("%H:%M") if timestamps else ""
                label = f"{start_t} ~ {end_t}" if start_t else f"window {idx}"
                line_start = chunk[0].get("line_start")
                line_end = chunk[-1].get("line_end")
                chunk_specs.append(
                    ChunkSpec(
                        text=text,
                        line_start=line_start,
                        line_end=line_end,
                        label=label,
                    )
                )
            chunks = _ensure_chunk_size(chunk_specs)
        else:
            if any(re.match(r"^\s{0,3}#{1,6}\s+", line) for line in normalized_raw.splitlines()):
                chunk_strategy = "markdown_heading"
                chunks = _chunk_markdown_by_heading(normalized_raw)
            else:
                chunk_strategy = "generic"
                chunks = _split_lines_into_chunks(normalized_raw.splitlines())
        normalized_text = normalized_raw
    elif suffix in STRUCTURED_SUFFIXES:
        chunk_strategy = "structured"
        normalized_text = _normalize_structured_text(normalized_raw, suffix)
        chunks = _chunk_structured_text(normalized_text)
    elif suffix in PYTHON_SUFFIXES:
        chunk_strategy = "python"
        normalized_text = normalized_raw
        chunks = _chunk_python_code(normalized_raw)
        symbols = _extract_python_symbols(normalized_raw, rel_path)
    else:
        chunk_strategy = "generic"
        normalized_text = normalized_raw
        chunks = _split_lines_into_chunks(normalized_text.splitlines())

    if not chunks:
        chunks = _split_lines_into_chunks(normalized_text.splitlines())
    original_chunk_total = len(chunks)
    chunks = _limit_chunk_specs(chunks, chunk_strategy=chunk_strategy)
    return normalized_text, chunks, symbols, chunk_strategy, max(1, len(normalized_text.splitlines())), original_chunk_total


def _chunks_for_indexed_text(normalized_text: str, suffix: str, chunk_strategy: str = "") -> list[ChunkSpec]:
    if not normalized_text:
        return []
    strategy = chunk_strategy.strip()
    if strategy == "wechat_markdown" or ((suffix in MARKDOWN_SUFFIXES or suffix in _BINARY_SUFFIXES) and not strategy):
        messages = _parse_md_messages_text(normalized_text) if suffix in MARKDOWN_SUFFIXES else []
        if messages:
            chunk_specs: list[ChunkSpec] = []
            for idx, chunk in enumerate(_chunk_by_time_window(messages), start=1):
                chunk_lines = []
                for message in chunk:
                    ts = message["timestamp"].strftime("%H:%M") if message.get("timestamp") else "?"
                    chunk_lines.append(f"[{ts}] {message['sender']}: {message['content']}")
                text = "\n".join(chunk_lines).strip()
                if not text:
                    continue
                timestamps = [item["timestamp"] for item in chunk if item.get("timestamp")]
                start_t = timestamps[0].strftime("%Y-%m-%d %H:%M") if timestamps else ""
                end_t = timestamps[-1].strftime("%H:%M") if timestamps else ""
                label = f"{start_t} ~ {end_t}" if start_t else f"window {idx}"
                line_start = chunk[0].get("line_start")
                line_end = chunk[-1].get("line_end")
                chunk_specs.append(
                    ChunkSpec(
                        text=text,
                        line_start=line_start,
                        line_end=line_end,
                        label=label,
                    )
                )
            chunks = _ensure_chunk_size(chunk_specs)
        else:
            chunks = _chunk_markdown_by_heading(normalized_text)
    elif strategy == "markdown_heading":
        chunks = _chunk_markdown_by_heading(normalized_text)
    elif strategy == "structured" or suffix in STRUCTURED_SUFFIXES:
        chunks = _chunk_structured_text(normalized_text)
    elif strategy == "python" or suffix in PYTHON_SUFFIXES:
        chunks = _chunk_python_code(normalized_text)
    else:
        chunks = _split_lines_into_chunks(normalized_text.splitlines())
    if chunks:
        return _limit_chunk_specs(chunks, chunk_strategy=strategy)
    return _limit_chunk_specs(_split_lines_into_chunks(normalized_text.splitlines()), chunk_strategy=strategy)


def _read_source_text(file_path: Path, suffix: str) -> str | None:
    try:
        if suffix in _BINARY_SUFFIXES:
            return _convert_binary_to_markdown(file_path)
        return file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def _normalized_cache_path(cache_root: Path, rel_path: str) -> Path:
    return cache_root / f"{rel_path}.txt"


def _process_source_file(root: Path, file_path: Path) -> IndexedFile | None:
    rel_path = _normalize_source_path(file_path.relative_to(root).as_posix())
    kb = _extract_kb(rel_path)
    suffix = file_path.suffix.lower()

    raw_text = _read_source_text(file_path, suffix)
    if raw_text is None:
        return None

    normalized_raw = _normalize_text(raw_text)
    if not normalized_raw:
        return None

    normalized_text, chunks, symbols, chunk_strategy, line_total, original_chunk_total = _prepare_indexed_content(
        rel_path,
        suffix,
        normalized_raw,
    )
    if not chunks:
        return None

    return IndexedFile(
        rel_path=rel_path,
        suffix=suffix,
        kb=kb,
        file_path=file_path,
        normalized_text=normalized_text,
        chunks=chunks,
        symbols=symbols,
        chunk_strategy=chunk_strategy,
        _original_chunk_total=original_chunk_total,
        _line_total=line_total,
    )


def _process_source_file_to_cache(root: Path, file_path: Path, cache_root: Path) -> IndexedFile | None:
    rel_path = _normalize_source_path(file_path.relative_to(root).as_posix())
    kb = _extract_kb(rel_path)
    suffix = file_path.suffix.lower()

    raw_text = _read_source_text(file_path, suffix)
    if raw_text is None:
        return None

    normalized_raw = _normalize_text(raw_text)
    if not normalized_raw:
        return None

    normalized_text, chunks, symbols, chunk_strategy, line_total, original_chunk_total = _prepare_indexed_content(
        rel_path,
        suffix,
        normalized_raw,
    )
    if not chunks:
        return None

    cache_path = _normalized_cache_path(cache_root, rel_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(normalized_text, encoding="utf-8")

    return IndexedFile(
        rel_path=rel_path,
        suffix=suffix,
        kb=kb,
        file_path=file_path,
        symbols=symbols,
        normalized_text_path=cache_path,
        chunk_strategy=chunk_strategy,
        _chunk_total=len(chunks),
        _original_chunk_total=original_chunk_total,
        _line_total=line_total,
    )


def _build_documents(source_dir: str) -> tuple[list[Document], list[IndexedFile]]:
    root = Path(source_dir)
    files = _iter_supported_files(root)
    documents: list[Document] = []
    indexed_files: list[IndexedFile] = []

    for file_path in files:
        processed = _process_source_file(root, file_path)
        if processed is None:
            continue

        indexed_files.append(processed)
        for idx, chunk in enumerate(processed.chunks):
            documents.append(
                Document(
                    page_content=chunk.text,
                    metadata={
                        "source": processed.rel_path,
                        "kb": processed.kb,
                        "chunk_index": idx,
                        "time_range": _chunk_location_label(chunk, f"chunk {idx + 1}"),
                        "line_start": chunk.line_start,
                        "line_end": chunk.line_end,
                    },
                )
            )

    return documents, indexed_files


_ORIGINAL_BUILD_DOCUMENTS = _build_documents


def _index_source_files(source_dir: str, cache_root: Path) -> tuple[list[IndexedFile], int]:
    root = Path(source_dir)
    indexed_files: list[IndexedFile] = []
    total_chunks = 0

    for file_path in _iter_supported_files(root):
        processed = _process_source_file_to_cache(root, file_path, cache_root)
        if processed is None:
            continue
        indexed_files.append(processed)
        total_chunks += processed.chunk_count

    return indexed_files, total_chunks


def _iter_documents_for_indexed_files(indexed_files: Iterable[IndexedFile]) -> Iterable[Document]:
    for indexed in indexed_files:
        for idx, chunk in enumerate(indexed.iter_chunks()):
            yield Document(
                page_content=chunk.text,
                metadata={
                    "source": indexed.rel_path,
                    "kb": indexed.kb,
                    "chunk_index": idx,
                    "time_range": _chunk_location_label(chunk, f"chunk {idx + 1}"),
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                },
            )


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


def _build_document_graph(indexed_files: list[IndexedFile]) -> dict[str, Any]:
    sources = sorted(
        {
            normalized
            for indexed in indexed_files
            if (normalized := _normalize_source_path(indexed.rel_path))
        }
    )
    source_lookup = set(sources)
    basename_lookup: dict[str, list[str]] = {}
    for source in sources:
        basename_lookup.setdefault(Path(source).name, []).append(source)

    edges_by_source: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {
        source: {} for source in sources
    }

    for indexed in indexed_files:
        source = _normalize_source_path(indexed.rel_path)
        if not source:
            continue
        for kind, reference in _iter_local_path_references_in_lines(indexed.iter_normalized_lines()):
            target = _resolve_document_reference(
                source,
                reference,
                source_lookup,
                basename_lookup,
            )
            if target is None:
                continue
            _add_graph_edge(edges_by_source, source, target, kind, reference)

    token_sources: dict[str, set[str]] = {}
    for indexed in indexed_files:
        source = _normalize_source_path(indexed.rel_path)
        if not source:
            continue
        for token in _extract_shared_tokens(indexed):
            token_sources.setdefault(token, set()).add(source)

    for token, matched_sources in token_sources.items():
        if not 2 <= len(matched_sources) <= GRAPH_SHARED_TOKEN_MAX_DOC_FREQ:
            continue
        ordered_sources = sorted(matched_sources)
        for source in ordered_sources:
            for target in ordered_sources:
                if source == target:
                    continue
                _add_graph_edge(edges_by_source, source, target, "shared_symbol", token)

    siblings_by_dir: dict[str, list[str]] = {}
    for source in sources:
        siblings_by_dir.setdefault(posixpath.dirname(source), []).append(source)
    for siblings in siblings_by_dir.values():
        for source in siblings:
            source_stem = Path(source).stem.lower()
            source_tokens = _stem_tokens(source_stem)
            for target in siblings:
                if source == target:
                    continue
                target_stem = Path(target).stem.lower()
                target_tokens = _stem_tokens(target_stem)
                shared_tokens = source_tokens & target_tokens
                if source_stem in target_stem or target_stem in source_stem or len(shared_tokens) >= 2:
                    _add_graph_edge(
                        edges_by_source,
                        source,
                        target,
                        "same_series",
                        ",".join(sorted(shared_tokens)) or target_stem,
                    )

    neighbors: dict[str, list[dict[str, Any]]] = {}
    edge_count = 0
    for source in sources:
        source_kb = _extract_kb(source)
        ranked = sorted(
            edges_by_source[source].values(),
            key=lambda item: (
                item[0],
                0 if _extract_kb(item[1]["target"]) == source_kb else 1,
                item[1]["target"],
            ),
        )
        selected = [payload for _, payload in ranked[:GRAPH_MAX_NEIGHBORS]]
        edge_count += len(selected)
        neighbors[source] = selected

    return {
        "version": 1,
        "edge_count": edge_count,
        "neighbors": neighbors,
    }


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


def _entity_semantic_node_id(node_type: str, name: str) -> str:
    normalized_type = str(node_type).strip().lower() or "concept"
    normalized_raw_name = name.strip().lower()
    normalized_name = re.sub(r"[^a-z0-9]+", "-", normalized_raw_name).strip("-")
    slug = normalized_name or "node"
    digest = hashlib.sha1(normalized_raw_name.encode("utf-8")).hexdigest()[:12]
    return f"{normalized_type}:{slug}:{digest}"


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


def _extract_section_reference_tokens(text: str) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for raw in CODE_SPAN_RE.findall(text):
        token = raw.strip()
        lowered = token.lower()
        if token and lowered not in seen:
            seen.add(lowered)
            tokens.append(token)
    for raw in CALL_RE.findall(text):
        token = raw.strip()
        lowered = token.lower()
        if token and lowered not in seen:
            seen.add(lowered)
            tokens.append(token)
    return tokens


def _build_symbol_reference_lookup(
    symbol_entries: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    seen_keys: set[tuple[str, str]] = set()
    for entry in symbol_entries:
        for raw in (entry.get("name", ""), entry.get("qualified_name", "")):
            key = str(raw).strip().lower()
            if not key:
                continue
            dedupe_key = (key, str(entry.get("id", "")))
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            lookup.setdefault(key, []).append(entry)
    return lookup


def _resolve_symbol_reference(
    source: str,
    token: str,
    lookup: dict[str, list[dict[str, Any]]],
) -> str | None:
    key = token.strip().lower()
    if not key:
        return None

    candidates = lookup.get(key, [])
    if not candidates:
        return None

    source = _normalize_source_path(source)
    source_kb = _extract_kb(source)

    def _prefer_specific(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        non_module = [entry for entry in entries if entry.get("symbol_kind") != "module"]
        return non_module or entries

    same_file = _prefer_specific([entry for entry in candidates if entry.get("source") == source])
    if len(same_file) == 1:
        return str(same_file[0]["id"])

    same_kb = _prefer_specific([entry for entry in candidates if entry.get("kb") == source_kb])
    if len(same_kb) == 1:
        return str(same_kb[0]["id"])

    candidates = _prefer_specific(candidates)
    if len(candidates) == 1:
        return str(candidates[0]["id"])
    return None


def _build_python_module_lookup(indexed_files: list[IndexedFile]) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for indexed in indexed_files:
        if indexed.suffix not in PYTHON_SUFFIXES:
            continue
        source = _normalize_source_path(indexed.rel_path)
        if not source:
            continue
        module_path = str(Path(source).with_suffix("")).replace("\\", "/")
        keys = {module_path.replace("/", "."), Path(source).stem}
        if Path(source).name == "__init__.py":
            package_name = posixpath.dirname(module_path).replace("/", ".")
            if package_name:
                keys.add(package_name)
        for key in keys:
            lowered = key.strip().lower()
            if not lowered:
                continue
            lookup.setdefault(lowered, []).append(source)
    return lookup


def _resolve_import_reference(
    source: str,
    qualified_name: str,
    module_lookup: dict[str, list[str]],
) -> str | None:
    qualified_name = qualified_name.strip().lower()
    if not qualified_name:
        return None

    source_kb = _extract_kb(source)
    parts = qualified_name.split(".")
    for end in range(len(parts), 0, -1):
        candidate = ".".join(parts[:end])
        matches = module_lookup.get(candidate, [])
        if not matches:
            continue
        same_kb = [match for match in matches if _extract_kb(match) == source_kb]
        if len(same_kb) == 1:
            return same_kb[0]
        if len(matches) == 1:
            return matches[0]
    return None


def _semantic_graph_enabled(
    llm_api_key: str,
    llm_model: str,
    llm_base_url: str,
) -> tuple[bool, str]:
    raw = os.getenv("SEMANTIC_GRAPH_ENABLED", "auto").strip().lower()
    enabled = bool(llm_api_key.strip() and llm_model.strip() and llm_base_url.strip())
    if raw in SEMANTIC_GRAPH_DISABLED_VALUES:
        return False, "disabled_by_env"
    if raw in SEMANTIC_GRAPH_ENABLED_VALUES:
        if enabled:
            return True, ""
        return False, "missing_llm_config"
    if enabled:
        return True, ""
    return False, "missing_llm_config"


def _semantic_section_fingerprint(
    source: str,
    chunk_index: int,
    text: str,
    llm_model: str,
    llm_base_url: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(source.encode("utf-8"))
    digest.update(f":{chunk_index}:".encode("utf-8"))
    digest.update(llm_model.encode("utf-8"))
    digest.update(llm_base_url.strip().rstrip("/").encode("utf-8"))
    digest.update(SEMANTIC_PROMPT_VERSION.encode("utf-8"))
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _semantic_cache_path(persist_path: Path) -> Path:
    return persist_path / SEMANTIC_EXTRACT_CACHE_FILENAME


def _load_semantic_cache(persist_path: Path) -> dict[str, Any]:
    path = _semantic_cache_path(persist_path)
    if not path.exists():
        return {"version": 1, "entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "entries": {}}
    if not isinstance(payload, dict):
        return {"version": 1, "entries": {}}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        payload["entries"] = {}
    return payload


def _write_semantic_cache(persist_path: Path, cache_payload: dict[str, Any]) -> None:
    _semantic_cache_path(persist_path).write_text(
        json.dumps(cache_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _semantic_aliases(name: str, aliases: Any) -> list[str]:
    values = [name]
    if isinstance(aliases, list):
        values.extend(str(item).strip() for item in aliases if str(item).strip())
    return _dedupe_strings(values)


def _register_semantic_aliases(
    alias_lookup: dict[str, list[str]],
    aliases: Iterable[str],
    node_id: str,
) -> None:
    for alias in aliases:
        lowered = str(alias).strip().lower()
        if not lowered:
            continue
        bucket = alias_lookup.setdefault(lowered, [])
        if node_id not in bucket:
            bucket.append(node_id)


def _query_notes_dir(persist_path: Path) -> Path:
    return persist_path / WIKI_DIRNAME / QUERY_NOTES_DIRNAME


def _extract_markdown_section(text: str, heading: str) -> str:
    pattern = re.compile(
        rf"^## {re.escape(heading)}\s*$\n?(.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return ""
    return match.group(1).strip()


def _unescape_markdown_link_text(text: str) -> str:
    return (
        text.replace("\\\\", "\\")
        .replace("\\[", "[")
        .replace("\\]", "]")
    )


def _parse_query_note_record(note_path: Path, persist_path: Path) -> dict[str, Any] | None:
    try:
        text = note_path.read_text(encoding="utf-8")
    except OSError:
        return None
    lines = text.splitlines()
    title = ""
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break
    if not title:
        return None

    note_relpath = note_path.relative_to(persist_path).as_posix()
    answer = _extract_markdown_section(text, "结论")
    sources_section = _extract_markdown_section(text, "来源")
    sources: list[str] = []
    for line in sources_section.splitlines():
        match = re.match(r"^- 来源文件：\[(.*)\]\([^)]+\)\s*$", line.strip())
        if not match:
            continue
        sources.append(_normalize_source_path(_unescape_markdown_link_text(match.group(1))))
    sources = _dedupe_strings(source for source in sources if source)

    created_at = ""
    for line in lines:
        if line.startswith("- 生成时间："):
            created_at = re.sub(r"^- 生成时间：`?(.*?)`?$", r"\1", line.strip())
            break

    tags: list[str] = []
    for line in lines:
        if not line.startswith("- 标签："):
            continue
        tags = re.findall(r"`([^`]+)`", line)
        break

    return {
        "id": _entity_query_note_node_id(note_relpath),
        "type": "query_note",
        "question": title,
        "name": title,
        "note_relpath": note_relpath,
        "summary": answer[:240].strip(),
        "created_at": created_at,
        "tags": tags,
        "sources": sources,
    }


def _load_query_note_records(persist_path: Path) -> list[dict[str, Any]]:
    query_dir = _query_notes_dir(persist_path)
    if not query_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for note_path in sorted(query_dir.rglob("*.md")):
        record = _parse_query_note_record(note_path, persist_path)
        if record is not None:
            records.append(record)
    return records


def _entity_attachment_files(
    nodes_by_id: dict[str, dict[str, Any]],
    normalized_edges: list[dict[str, Any]],
    attachment_types: set[str] | None = None,
) -> dict[str, set[str]]:
    selected_types = attachment_types or ENTITY_ATTACHMENT_NODE_TYPES
    attachment_node_ids = {
        node_id
        for node_id, node in nodes_by_id.items()
        if str(node.get("type", "") or "") in selected_types
    }
    attachment_files: dict[str, set[str]] = {node_id: set() for node_id in attachment_node_ids}
    for normalized_edge in normalized_edges:
        source_id = str(normalized_edge.get("source", "") or "")
        target_id = str(normalized_edge.get("target", "") or "")
        source_node = nodes_by_id.get(source_id)
        target_node = nodes_by_id.get(target_id)
        if not source_node or not target_node:
            continue
        if source_id in attachment_node_ids:
            target_file = _entity_node_file_source(target_node)
            if target_file:
                attachment_files.setdefault(source_id, set()).add(target_file)
        if target_id in attachment_node_ids:
            source_file = _entity_node_file_source(source_node)
            if source_file:
                attachment_files.setdefault(target_id, set()).add(source_file)
    return attachment_files


def _entity_edge_evidence_source(edge: dict[str, Any]) -> str:
    evidence = edge.get("evidence")
    if not isinstance(evidence, dict):
        return ""
    return _normalize_source_path(str(evidence.get("source", "") or ""))


def _build_file_only_entity_graph(
    file_sources: list[str],
    document_graph: dict[str, Any],
) -> dict[str, Any]:
    normalized_sources = sorted(
        {
            _normalize_source_path(source)
            for source in file_sources
            if _normalize_source_path(source)
        }
    )
    nodes: list[dict[str, Any]] = []
    node_seen: set[str] = set()
    edges: dict[tuple[Any, ...], dict[str, Any]] = {}
    for source in normalized_sources:
        _add_entity_node(
            nodes,
            node_seen,
            {
                "id": _entity_file_node_id(source),
                "type": "file",
                "name": Path(source).name,
                "source": source,
                "path": source,
                "kb": _extract_kb(source),
                "line_start": 1,
                "line_end": 1,
                "confidence": "EXTRACTED",
            },
        )

    raw_neighbors = document_graph.get("neighbors", {})
    if isinstance(raw_neighbors, dict):
        for source, raw_edges in raw_neighbors.items():
            normalized_source = _normalize_source_path(source)
            if not normalized_source or not isinstance(raw_edges, list):
                continue
            for edge in raw_edges:
                if not isinstance(edge, dict):
                    continue
                target = _normalize_source_path(edge.get("target", ""))
                if not target:
                    continue
                _add_entity_edge(
                    edges,
                    _entity_file_node_id(normalized_source),
                    _entity_file_node_id(target),
                    str(edge.get("kind") or ""),
                    evidence_source=normalized_source,
                    reason=str(edge.get("reason") or ""),
                )

    sorted_nodes = sorted(
        nodes,
        key=lambda item: (
            str(item.get("type", "")),
            str(item.get("source", "")),
            str(item.get("name", "")),
        ),
    )
    sorted_edges = sorted(
        edges.values(),
        key=lambda item: (
            str(item.get("source", "")),
            str(item.get("target", "")),
            str(item.get("type", "")),
            str(item.get("reason", "")),
        ),
    )
    return {
        "version": 1,
        "node_count": len(sorted_nodes),
        "edge_count": len(sorted_edges),
        "nodes": sorted_nodes,
        "edges": sorted_edges,
    }


def _merge_query_notes_into_entity_graph(
    entity_graph: dict[str, Any],
    query_notes: list[dict[str, Any]],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    node_seen: set[str] = set()
    nodes_by_id: dict[str, dict[str, Any]] = {}
    query_note_ids: set[str] = set()
    for raw_node in entity_graph.get("nodes", []):
        if not isinstance(raw_node, dict):
            continue
        normalized_node = _normalize_entity_node_payload(raw_node)
        if normalized_node is None:
            continue
        if str(normalized_node.get("type", "") or "") == "query_note":
            query_note_ids.add(str(normalized_node["id"]))
            continue
        _add_entity_node(nodes, node_seen, normalized_node)
        nodes_by_id[str(normalized_node["id"])] = normalized_node

    edges: dict[tuple[Any, ...], dict[str, Any]] = {}
    normalized_edges: list[dict[str, Any]] = []
    for raw_edge in entity_graph.get("edges", []):
        if not isinstance(raw_edge, dict):
            continue
        normalized_edge = _normalize_entity_edge_payload(raw_edge)
        if normalized_edge is None:
            continue
        source_id = str(normalized_edge.get("source", "") or "")
        target_id = str(normalized_edge.get("target", "") or "")
        if source_id in query_note_ids or target_id in query_note_ids:
            continue
        normalized_edges.append(normalized_edge)
        evidence = normalized_edge.get("evidence")
        evidence_source = ""
        line_start = None
        line_end = None
        if isinstance(evidence, dict):
            evidence_source = _normalize_source_path(str(evidence.get("source", "") or ""))
            line_start = evidence.get("line_start")
            line_end = evidence.get("line_end")
        if not evidence_source:
            evidence_source = _normalize_source_path(str(normalized_edge.get("reason", "") or ""))
        if not evidence_source:
            evidence_source = "graph"
        _add_entity_edge(
            edges,
            source_id=source_id,
            target_id=target_id,
            kind=str(normalized_edge.get("type", "") or ""),
            evidence_source=evidence_source,
            reason=str(normalized_edge.get("reason", "") or ""),
            line_start=line_start if isinstance(line_start, int) else None,
            line_end=line_end if isinstance(line_end, int) else None,
        )

    semantic_attachment_files = _entity_attachment_files(
        nodes_by_id,
        normalized_edges,
        attachment_types={"concept", "decision"},
    )

    for record in query_notes:
        note_id = str(record.get("id", "") or "").strip()
        question = str(record.get("question", "") or "").strip()
        note_relpath = _normalize_source_path(str(record.get("note_relpath", "") or ""))
        if not note_id or not question or not note_relpath:
            continue
        payload = {
            "id": note_id,
            "type": "query_note",
            "name": question,
            "summary": str(record.get("summary", "") or "").strip(),
            "note_relpath": note_relpath,
            "created_at": str(record.get("created_at", "") or ""),
            "tags": [str(tag) for tag in record.get("tags", []) if str(tag).strip()],
            "confidence": "EXTRACTED",
        }
        _add_entity_node(nodes, node_seen, payload)
        nodes_by_id[note_id] = payload
        for source in record.get("sources", []):
            normalized_source = _normalize_source_path(source)
            file_node_id = _entity_file_node_id(normalized_source)
            if file_node_id not in nodes_by_id:
                continue
            _add_entity_edge(
                edges,
                source_id=file_node_id,
                target_id=note_id,
                kind="semantically_related",
                evidence_source=note_relpath,
                reason=question,
            )
            _add_entity_edge(
                edges,
                source_id=note_id,
                target_id=file_node_id,
                kind="semantically_related",
                evidence_source=note_relpath,
                reason=question,
            )
            for semantic_node_id, attached_files in semantic_attachment_files.items():
                if normalized_source not in attached_files:
                    continue
                semantic_node = nodes_by_id.get(semantic_node_id)
                if semantic_node is None:
                    continue
                relation = str(semantic_node.get("type", "") or "semantically_related")
                _add_entity_edge(
                    edges,
                    source_id=note_id,
                    target_id=semantic_node_id,
                    kind="semantically_related",
                    evidence_source=normalized_source,
                    reason=question or relation,
                )
                _add_entity_edge(
                    edges,
                    source_id=semantic_node_id,
                    target_id=note_id,
                    kind="semantically_related",
                    evidence_source=normalized_source,
                    reason=question or relation,
                )

    sorted_nodes = sorted(
        nodes,
        key=lambda item: (
            str(item.get("type", "")),
            str(item.get("source", "")),
            str(item.get("note_relpath", "")),
            int(item.get("chunk_index") or 0),
            int(item.get("line_start") or 0),
            str(item.get("qualified_name") or item.get("name") or ""),
        ),
    )
    sorted_edges = sorted(
        edges.values(),
        key=lambda item: (
            str(item.get("source", "")),
            str(item.get("target", "")),
            str(item.get("type", "")),
            str(item.get("reason", "")),
        ),
    )
    return {
        **entity_graph,
        "nodes": sorted_nodes,
        "edges": sorted_edges,
        "node_count": len(sorted_nodes),
        "edge_count": len(sorted_edges),
    }


def _semantic_extraction_prompt(section: dict[str, Any]) -> str:
    text = str(section.get("text", "") or "").strip()
    if len(text) > SEMANTIC_SECTION_MAX_CHARS:
        text = text[: SEMANTIC_SECTION_MAX_CHARS - 3].rstrip() + "..."
    source = str(section.get("source", "") or "")
    label = str(section.get("label", "") or "")
    return f"""你是知识图谱抽取器。请只根据给定片段抽取明确出现或被明确陈述的语义概念和决策。

只返回一个 JSON 对象，不要输出任何额外说明，结构必须是：
{{
  "concepts": [
    {{
      "name": "概念名",
      "summary": "一句话说明",
      "aliases": ["可选别名"]
    }}
  ],
  "decisions": [
    {{
      "name": "决策名",
      "summary": "一句话说明",
      "aliases": ["可选别名"],
      "rationale": ["支撑该决策的概念、理由或依据短语"]
    }}
  ]
}}

要求：
1. 只有文本里明确表达的抽象概念、业务术语、系统名称、架构决策才能抽取。
2. `summary` 必须简短，不要重复原文大段句子。
3. 如果没有可抽取内容，返回空数组。
4. `rationale` 只保留能直接从文本看出的依据短语。

片段来源：{source}
片段标签：{label}

片段正文：
\"\"\"
{text}
\"\"\"
"""


def _response_token_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        response_metadata = getattr(response, "response_metadata", None)
        if isinstance(response_metadata, dict):
            raw_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
            if isinstance(raw_usage, dict):
                usage = raw_usage
    if not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    prompt_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _normalize_semantic_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    name = str(item.get("name", "") or "").strip()
    if not name:
        return None
    aliases = _semantic_aliases(name, item.get("aliases"))
    payload: dict[str, Any] = {
        "name": name,
        "aliases": aliases,
    }
    summary = str(item.get("summary", "") or "").strip()
    if summary:
        payload["summary"] = summary
    rationale = item.get("rationale")
    if isinstance(rationale, list):
        payload["rationale"] = _dedupe_strings(str(entry) for entry in rationale if str(entry).strip())
    return payload


def _parse_semantic_payload(content: str) -> dict[str, list[dict[str, Any]]]:
    payload = _extract_json_blob(content)
    concepts = payload.get("concepts") or []
    decisions = payload.get("decisions") or []
    return {
        "concepts": [item for item in (_normalize_semantic_item(entry) for entry in concepts) if item],
        "decisions": [item for item in (_normalize_semantic_item(entry) for entry in decisions) if item],
    }


def _upsert_semantic_node(
    nodes_by_id: dict[str, dict[str, Any]],
    nodes: list[dict[str, Any]],
    node_seen: set[str],
    alias_lookup: dict[str, list[str]],
    node_type: str,
    item: dict[str, Any],
) -> str:
    name = str(item.get("name", "") or "").strip()
    node_id = _entity_semantic_node_id(node_type, name)
    aliases = _semantic_aliases(name, item.get("aliases"))
    if node_id not in nodes_by_id:
        payload = {
            "id": node_id,
            "type": node_type,
            "name": name,
            "aliases": aliases,
            "summary": str(item.get("summary", "") or "").strip(),
            "confidence": "INFERRED",
        }
        _add_entity_node(nodes, node_seen, payload)
        nodes_by_id[node_id] = payload
    else:
        payload = nodes_by_id[node_id]
        existing_aliases = payload.get("aliases", [])
        merged_aliases = _dedupe_strings(existing_aliases if isinstance(existing_aliases, list) else [])
        for alias in aliases:
            if alias.lower() not in {item.lower() for item in merged_aliases}:
                merged_aliases.append(alias)
        payload["aliases"] = merged_aliases
        if not str(payload.get("summary", "") or "").strip():
            summary = str(item.get("summary", "") or "").strip()
            if summary:
                payload["summary"] = summary
    _register_semantic_aliases(alias_lookup, aliases, node_id)
    return node_id


def _iter_semantic_sections(indexed_files: list[IndexedFile]) -> Iterable[dict[str, Any]]:
    for indexed in indexed_files:
        source = _normalize_source_path(indexed.rel_path)
        if not source:
            continue
        for chunk_index, chunk in enumerate(indexed.iter_chunks()):
            text = chunk.text.strip()
            if not text:
                continue
            yield (
                {
                    "source": source,
                    "kb": indexed.kb,
                    "suffix": indexed.suffix,
                    "chunk_index": chunk_index,
                    "label": _chunk_location_label(chunk, f"chunk {chunk_index + 1}"),
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                    "text": text,
                }
            )


def _extract_semantic_sections(
    indexed_files: list[IndexedFile],
    persist_path: Path,
    llm_api_key: str,
    llm_model: str,
    llm_base_url: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_llm_base_url = llm_base_url.strip().rstrip("/")
    enabled, disabled_reason = _semantic_graph_enabled(
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
    )
    stats: dict[str, Any] = {
        "enabled": enabled,
        "disabled_reason": disabled_reason,
        "llm_model": llm_model,
        "prompt_version": SEMANTIC_PROMPT_VERSION,
        "sections_total": 0,
        "cached_sections": 0,
        "extracted_sections": 0,
        "failed_sections": 0,
        "api_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "duration_seconds": 0.0,
    }
    stats["sections_total"] = sum(indexed.chunk_count for indexed in indexed_files)
    if not enabled:
        return [], stats

    started = time.perf_counter()
    llm = make_llm(
        api_key=llm_api_key,
        model=llm_model,
        base_url=llm_base_url,
        temperature=0.0,
    )
    cache = _load_semantic_cache(persist_path)
    previous_entries = cache.get("entries", {})
    if not isinstance(previous_entries, dict):
        previous_entries = {}
    next_entries: dict[str, Any] = {}
    extracted_sections: list[dict[str, Any]] = []

    for section in _iter_semantic_sections(indexed_files):
        cache_key = _entity_section_node_id(str(section["source"]), int(section["chunk_index"]))
        fingerprint = _semantic_section_fingerprint(
            source=str(section["source"]),
            chunk_index=int(section["chunk_index"]),
            text=str(section["text"]),
            llm_model=llm_model,
            llm_base_url=normalized_llm_base_url,
        )
        cached_entry = previous_entries.get(cache_key)
        if (
            isinstance(cached_entry, dict)
            and cached_entry.get("fingerprint") == fingerprint
            and cached_entry.get("prompt_version") == SEMANTIC_PROMPT_VERSION
            and cached_entry.get("llm_model") == llm_model
            and str(cached_entry.get("llm_base_url", "") or "") == normalized_llm_base_url
            and cached_entry.get("status") == "ok"
        ):
            payload = cached_entry.get("payload", {})
            if not isinstance(payload, dict):
                payload = {"concepts": [], "decisions": []}
            stats["cached_sections"] += 1
            next_entries[cache_key] = cached_entry
        else:
            try:
                response = llm.invoke(_semantic_extraction_prompt(section))
                payload = _parse_semantic_payload(_chunk_to_text(response))
                usage = _response_token_usage(response)
                stats["api_calls"] += 1
                stats["extracted_sections"] += 1
                stats["prompt_tokens"] += usage["prompt_tokens"]
                stats["completion_tokens"] += usage["completion_tokens"]
                stats["total_tokens"] += usage["total_tokens"]
                next_entries[cache_key] = {
                    "status": "ok",
                    "fingerprint": fingerprint,
                    "prompt_version": SEMANTIC_PROMPT_VERSION,
                    "llm_model": llm_model,
                    "llm_base_url": normalized_llm_base_url,
                    "payload": payload,
                    "usage": usage,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
            except Exception as exc:
                payload = {"concepts": [], "decisions": []}
                stats["failed_sections"] += 1
                next_entries[cache_key] = {
                    "status": "error",
                    "fingerprint": fingerprint,
                    "prompt_version": SEMANTIC_PROMPT_VERSION,
                    "llm_model": llm_model,
                    "llm_base_url": normalized_llm_base_url,
                    "payload": payload,
                    "error": str(exc),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
        if payload.get("concepts") or payload.get("decisions"):
            extracted_sections.append(
                {
                    "source": str(section["source"]),
                    "chunk_index": int(section["chunk_index"]),
                    "label": str(section["label"]),
                    "line_start": section.get("line_start"),
                    "line_end": section.get("line_end"),
                    "payload": payload,
                }
            )

    stats["duration_seconds"] = round(time.perf_counter() - started, 3)
    _write_semantic_cache(
        persist_path,
        {
            "version": 1,
            "prompt_version": SEMANTIC_PROMPT_VERSION,
            "llm_model": llm_model,
            "llm_base_url": normalized_llm_base_url,
            "entries": next_entries,
        },
    )
    return extracted_sections, stats


def _build_entity_graph(
    indexed_files: list[IndexedFile],
    document_graph: dict[str, Any],
    semantic_sections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    node_seen: set[str] = set()
    nodes_by_id: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[Any, ...], dict[str, Any]] = {}

    normalized_files: list[tuple[str, IndexedFile]] = []
    source_lookup: set[str] = set()
    basename_lookup: dict[str, list[str]] = {}
    for indexed in indexed_files:
        source = _normalize_source_path(indexed.rel_path)
        if not source:
            continue
        normalized_files.append((source, indexed))
        source_lookup.add(source)
        basename_lookup.setdefault(Path(source).name, []).append(source)

    section_records_by_source: dict[str, list[dict[str, Any]]] = {}
    section_records_by_id: dict[str, dict[str, Any]] = {}
    symbol_records_by_source: dict[str, list[dict[str, Any]]] = {}
    all_symbol_entries: list[dict[str, Any]] = []
    pending_rationale_edges: list[dict[str, Any]] = []

    for source, indexed in sorted(normalized_files, key=lambda item: item[0]):
        total_lines = indexed.line_count
        kb = _extract_kb(source) or indexed.kb
        _add_entity_node(
            nodes,
            node_seen,
            {
                "id": _entity_file_node_id(source),
                "type": "file",
                "name": Path(source).name,
                "source": source,
                "path": source,
                "kb": kb,
                "line_start": 1,
                "line_end": total_lines,
                "confidence": "EXTRACTED",
            },
        )
        nodes_by_id[_entity_file_node_id(source)] = nodes[-1]

        section_records: list[dict[str, Any]] = []
        for chunk_index, chunk in enumerate(indexed.iter_chunks()):
            label = chunk.label or f"chunk {chunk_index + 1}"
            section_id = _entity_section_node_id(source, chunk_index)
            section_line_start = chunk.line_start or 1
            section_line_end = chunk.line_end or section_line_start
            section_record = {
                "id": section_id,
                "type": "section",
                "name": label,
                "source": source,
                "file": source,
                "kb": kb,
                "chunk_index": chunk_index,
                "line_start": section_line_start,
                "line_end": section_line_end,
                "confidence": "EXTRACTED",
            }
            _add_entity_node(nodes, node_seen, section_record)
            nodes_by_id[section_id] = nodes[-1]
            _add_entity_edge(
                edges,
                _entity_file_node_id(source),
                section_id,
                "contains",
                evidence_source=source,
                reason=label,
                line_start=section_line_start,
                line_end=section_line_end,
            )
            enriched_section_record = {**section_record, "chunk": chunk}
            section_records.append(enriched_section_record)
            section_records_by_id[section_id] = enriched_section_record
        section_records_by_source[source] = section_records

        symbol_records: list[dict[str, Any]] = []
        for record in indexed.symbols:
            symbol_source = _normalize_source_path(record.get("source", source)) or source
            symbol_line_start = int(record.get("line_start") or 1)
            symbol_line_end = int(record.get("line_end") or symbol_line_start)
            symbol_record = {
                "id": _entity_symbol_node_id(record),
                "type": "symbol",
                "name": str(record.get("name") or ""),
                "qualified_name": str(record.get("qualified_name") or record.get("name") or ""),
                "symbol_kind": str(record.get("kind") or "symbol"),
                "signature": str(record.get("signature") or ""),
                "source": symbol_source,
                "file": symbol_source,
                "kb": _extract_kb(symbol_source) or kb,
                "line_start": symbol_line_start,
                "line_end": symbol_line_end,
                "confidence": "EXTRACTED",
            }
            _add_entity_node(nodes, node_seen, symbol_record)
            nodes_by_id[str(symbol_record["id"])] = nodes[-1]
            _add_entity_edge(
                edges,
                _entity_file_node_id(source),
                str(symbol_record["id"]),
                "defines",
                evidence_source=source,
                reason=str(symbol_record["qualified_name"]),
                line_start=symbol_line_start,
                line_end=symbol_line_end,
            )
            symbol_records.append(symbol_record)
            all_symbol_entries.append(symbol_record)
        symbol_records_by_source[source] = symbol_records

    for source, section_records in section_records_by_source.items():
        symbol_records = symbol_records_by_source.get(source, [])
        for section_record in section_records:
            section_start = int(section_record["line_start"])
            section_end = int(section_record["line_end"])
            for symbol_record in symbol_records:
                if symbol_record.get("symbol_kind") == "module":
                    continue
                symbol_start = int(symbol_record["line_start"])
                if section_start <= symbol_start <= section_end:
                    _add_entity_edge(
                        edges,
                        str(section_record["id"]),
                        str(symbol_record["id"]),
                        "contains",
                        evidence_source=source,
                        reason=str(section_record["name"]),
                        line_start=section_start,
                        line_end=section_end,
                    )

    symbol_lookup = _build_symbol_reference_lookup(all_symbol_entries)
    module_lookup = _build_python_module_lookup(indexed_files)

    for source, section_records in section_records_by_source.items():
        for section_record in section_records:
            chunk = section_record["chunk"]
            section_id = str(section_record["id"])
            line_start = int(section_record["line_start"])
            line_end = int(section_record["line_end"])
            for kind, reference in _iter_local_path_references(chunk.text):
                target = _resolve_document_reference(
                    source,
                    reference,
                    source_lookup,
                    basename_lookup,
                )
                if target is None:
                    continue
                _add_entity_edge(
                    edges,
                    section_id,
                    _entity_file_node_id(target),
                    kind,
                    evidence_source=source,
                    reason=reference,
                    line_start=line_start,
                    line_end=line_end,
                )

            for token in _extract_section_reference_tokens(chunk.text):
                target_symbol_id = _resolve_symbol_reference(source, token, symbol_lookup)
                if target_symbol_id is None:
                    continue
                _add_entity_edge(
                    edges,
                    section_id,
                    target_symbol_id,
                    "references",
                    evidence_source=source,
                    reason=token,
                    line_start=line_start,
                    line_end=line_end,
                )

    for symbol_record in all_symbol_entries:
        if symbol_record.get("symbol_kind") != "import":
            continue
        target = _resolve_import_reference(
            str(symbol_record["source"]),
            str(symbol_record.get("qualified_name") or ""),
            module_lookup,
        )
        if target is None:
            continue
        _add_entity_edge(
            edges,
            str(symbol_record["id"]),
            _entity_file_node_id(target),
            "imports",
            evidence_source=str(symbol_record["source"]),
            reason=str(symbol_record.get("qualified_name") or ""),
            line_start=int(symbol_record["line_start"]),
            line_end=int(symbol_record["line_end"]),
        )

    concept_alias_lookup: dict[str, list[str]] = {}
    decision_alias_lookup: dict[str, list[str]] = {}
    semantic_sections = semantic_sections or []
    for semantic_section in semantic_sections:
        source = _normalize_source_path(semantic_section.get("source", ""))
        chunk_index = int(semantic_section.get("chunk_index") or 0)
        section_id = _entity_section_node_id(source, chunk_index)
        section_record = section_records_by_id.get(section_id)
        if not source or section_record is None:
            continue
        file_node_id = _entity_file_node_id(source)
        line_start = int(section_record.get("line_start") or 1)
        line_end = int(section_record.get("line_end") or line_start)
        payload = semantic_section.get("payload")
        if not isinstance(payload, dict):
            continue

        for concept in payload.get("concepts", []):
            if not isinstance(concept, dict):
                continue
            concept_id = _upsert_semantic_node(
                nodes_by_id=nodes_by_id,
                nodes=nodes,
                node_seen=node_seen,
                alias_lookup=concept_alias_lookup,
                node_type="concept",
                item=concept,
            )
            reason = str(concept.get("name") or "")
            _add_entity_edge(
                edges,
                section_id,
                concept_id,
                "semantically_related",
                evidence_source=source,
                reason=reason,
                line_start=line_start,
                line_end=line_end,
            )
            _add_entity_edge(
                edges,
                file_node_id,
                concept_id,
                "semantically_related",
                evidence_source=source,
                reason=reason,
                line_start=line_start,
                line_end=line_end,
            )
            _add_entity_edge(
                edges,
                concept_id,
                file_node_id,
                "semantically_related",
                evidence_source=source,
                reason=reason,
                line_start=line_start,
                line_end=line_end,
            )

        for decision in payload.get("decisions", []):
            if not isinstance(decision, dict):
                continue
            decision_id = _upsert_semantic_node(
                nodes_by_id=nodes_by_id,
                nodes=nodes,
                node_seen=node_seen,
                alias_lookup=decision_alias_lookup,
                node_type="decision",
                item=decision,
            )
            reason = str(decision.get("name") or "")
            _add_entity_edge(
                edges,
                section_id,
                decision_id,
                "semantically_related",
                evidence_source=source,
                reason=reason,
                line_start=line_start,
                line_end=line_end,
            )
            _add_entity_edge(
                edges,
                file_node_id,
                decision_id,
                "semantically_related",
                evidence_source=source,
                reason=reason,
                line_start=line_start,
                line_end=line_end,
            )
            _add_entity_edge(
                edges,
                decision_id,
                file_node_id,
                "semantically_related",
                evidence_source=source,
                reason=reason,
                line_start=line_start,
                line_end=line_end,
            )
            rationale_values = decision.get("rationale")
            if isinstance(rationale_values, list):
                pending_rationale_edges.append(
                    {
                        "decision_id": decision_id,
                        "rationale": list(rationale_values),
                        "source": source,
                        "line_start": line_start,
                        "line_end": line_end,
                    }
                )

    for pending_edge in pending_rationale_edges:
        decision_id = str(pending_edge.get("decision_id", "") or "")
        source = str(pending_edge.get("source", "") or "")
        line_start = int(pending_edge.get("line_start") or 1)
        line_end = int(pending_edge.get("line_end") or line_start)
        rationale_values = pending_edge.get("rationale")
        if not decision_id or not isinstance(rationale_values, list):
            continue
        for rationale in rationale_values:
            rationale_key = str(rationale or "").strip().lower()
            if not rationale_key:
                continue
            concept_ids = concept_alias_lookup.get(rationale_key, [])
            if not concept_ids:
                continue
            for concept_id in sorted(set(concept_ids)):
                _add_entity_edge(
                    edges,
                    concept_id,
                    decision_id,
                    "rationale_for",
                    evidence_source=source,
                    reason=str(rationale),
                    line_start=line_start,
                    line_end=line_end,
                )

    raw_neighbors = document_graph.get("neighbors", {})
    if isinstance(raw_neighbors, dict):
        for source, raw_edges in raw_neighbors.items():
            normalized_source = _normalize_source_path(source)
            if not normalized_source or not isinstance(raw_edges, list):
                continue
            for edge in raw_edges:
                if not isinstance(edge, dict):
                    continue
                target = _normalize_source_path(edge.get("target", ""))
                if not target:
                    continue
                _add_entity_edge(
                    edges,
                    _entity_file_node_id(normalized_source),
                    _entity_file_node_id(target),
                    str(edge.get("kind") or ""),
                    evidence_source=normalized_source,
                    reason=str(edge.get("reason") or ""),
                )

    sorted_nodes = sorted(
        nodes,
        key=lambda item: (
            str(item.get("type", "")),
            str(item.get("source", "")),
            int(item.get("chunk_index") or 0),
            int(item.get("line_start") or 0),
            str(item.get("qualified_name") or item.get("name") or ""),
        ),
    )
    sorted_edges = sorted(
        edges.values(),
        key=lambda item: (
            str(item.get("source", "")),
            str(item.get("target", "")),
            str(item.get("type", "")),
            str(item.get("reason", "")),
        ),
    )
    return {
        "version": 1,
        "node_count": len(sorted_nodes),
        "edge_count": len(sorted_edges),
        "nodes": sorted_nodes,
        "edges": sorted_edges,
    }


def _normalize_entity_node_id(node_id: str) -> str:
    text = str(node_id).strip()
    if not text:
        return ""
    if text.startswith("file:"):
        return f"file:{_normalize_source_path(text[5:])}"
    if text.startswith("section:"):
        body = text[8:]
        source, sep, chunk_index = body.rpartition(":")
        if sep and source:
            return f"section:{_normalize_source_path(source)}:{chunk_index}"
    if text.startswith("symbol:"):
        body = text[7:]
        parts = body.rsplit(":", 3)
        if len(parts) == 4:
            source, symbol_kind, qualified_name, line_start = parts
            return f"symbol:{_normalize_source_path(source)}:{symbol_kind}:{qualified_name}:{line_start}"
        if len(parts) == 3:
            source, qualified_name, line_start = parts
            return f"symbol:{_normalize_source_path(source)}:{qualified_name}:{line_start}"
    return text


def _normalize_entity_node_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    node_id = _normalize_entity_node_id(str(payload.get("id", "")))
    if not node_id:
        return None

    normalized = dict(payload)
    normalized["id"] = node_id
    for key in ("source", "file", "path"):
        value = normalized.get(key)
        if isinstance(value, str):
            clean = _normalize_source_path(value)
            if clean:
                normalized[key] = clean
            elif key in normalized:
                normalized.pop(key)

    source = str(normalized.get("source", "") or "")
    if source and not normalized.get("kb"):
        normalized["kb"] = _extract_kb(source)
    return normalized


def _normalize_entity_edge_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    source = _normalize_entity_node_id(str(payload.get("source", "")))
    target = _normalize_entity_node_id(str(payload.get("target", "")))
    if not source or not target or source == target:
        return None

    normalized = dict(payload)
    normalized["source"] = source
    normalized["target"] = target

    evidence = normalized.get("evidence")
    if isinstance(evidence, dict):
        normalized_evidence = dict(evidence)
        evidence_source = normalized_evidence.get("source")
        if isinstance(evidence_source, str):
            clean = _normalize_source_path(evidence_source)
            if clean:
                normalized_evidence["source"] = clean
            else:
                normalized_evidence.pop("source", None)
        normalized["evidence"] = normalized_evidence
    return normalized


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


def _project_file_relationships(
    file_sources: list[str],
    document_graph: dict[str, Any],
    entity_graph: dict[str, Any],
) -> list[dict[str, Any]]:
    file_sources = sorted({_normalize_source_path(source) for source in file_sources if _normalize_source_path(source)})
    valid_sources = set(file_sources)
    relationships: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def add(
        source: str,
        target: str,
        kind: str,
        reason: str,
        origin: str,
        bridges: list[dict[str, Any]] | None = None,
    ) -> None:
        source = _normalize_source_path(source)
        target = _normalize_source_path(target)
        if not source or not target or source == target:
            return
        if source not in valid_sources or target not in valid_sources:
            return
        bridge_entities = bridges or []
        key = (
            source,
            target,
            kind,
            reason,
            origin,
            tuple((item.get("id", ""), item.get("relation", "")) for item in bridge_entities),
        )
        if key in seen:
            return
        seen.add(key)
        relationships.append(
            {
                "source": source,
                "target": target,
                "kind": kind,
                "reason": reason,
                "origin": origin,
                "bridges": bridge_entities,
            }
        )

    raw_neighbors = document_graph.get("neighbors", {})
    if isinstance(raw_neighbors, dict):
        for source, edges in raw_neighbors.items():
            normalized_source = _normalize_source_path(source)
            if not normalized_source or not isinstance(edges, list):
                continue
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                add(
                    source=normalized_source,
                    target=str(edge.get("target", "") or ""),
                    kind=str(edge.get("kind", "") or ""),
                    reason=str(edge.get("reason", "") or ""),
                    origin="document_graph",
                )

    entity_nodes_by_id: dict[str, dict[str, Any]] = {}
    raw_nodes = entity_graph.get("nodes", [])
    if isinstance(raw_nodes, list):
        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict):
                continue
            normalized_node = _normalize_entity_node_payload(raw_node)
            if normalized_node is None:
                continue
            entity_nodes_by_id[str(normalized_node["id"])] = normalized_node

    normalized_entity_edges: list[dict[str, Any]] = []
    raw_edges = entity_graph.get("edges", [])
    if isinstance(raw_edges, list):
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                continue
            normalized_edge = _normalize_entity_edge_payload(raw_edge)
            if normalized_edge is None:
                continue
            normalized_entity_edges.append(normalized_edge)
    semantic_attachment_files = _entity_attachment_files(
        entity_nodes_by_id,
        normalized_entity_edges,
    )
    attachment_node_ids = set(semantic_attachment_files.keys())

    for normalized_edge in normalized_entity_edges:
        source_id = str(normalized_edge["source"])
        target_id = str(normalized_edge["target"])
        source_node = entity_nodes_by_id.get(source_id)
        target_node = entity_nodes_by_id.get(target_id)
        if not source_node or not target_node:
            continue
        source_files = set()
        target_files = set()
        source_file = _entity_node_file_source(source_node)
        target_file = _entity_node_file_source(target_node)
        evidence_source = _entity_edge_evidence_source(normalized_edge)
        if source_file:
            source_files.add(source_file)
        elif source_id in attachment_node_ids:
            source_files |= semantic_attachment_files.get(source_id, set())
            if (
                str(source_node.get("type", "") or "") == "query_note"
                and evidence_source
                and evidence_source in source_files
            ):
                source_files = {evidence_source}
        if target_file:
            target_files.add(target_file)
        elif target_id in attachment_node_ids:
            target_files |= semantic_attachment_files.get(target_id, set())
            if (
                str(target_node.get("type", "") or "") == "query_note"
                and evidence_source
                and evidence_source in target_files
            ):
                target_files = {evidence_source}
        if not source_files or not target_files:
            continue
        bridges: list[dict[str, Any]] = []
        if source_node.get("type") != "file":
            bridges.append(
                _entity_bridge_payload(
                    source_node,
                    relation=str(normalized_edge.get("type", "") or ""),
                )
            )
        if target_node.get("type") != "file":
            bridges.append(
                _entity_bridge_payload(
                    target_node,
                    relation=str(normalized_edge.get("type", "") or ""),
                )
            )
        for projected_source in sorted(source_files):
            for projected_target in sorted(target_files):
                if projected_source == projected_target:
                    continue
                add(
                    source=projected_source,
                    target=projected_target,
                    kind=str(normalized_edge.get("type", "") or ""),
                    reason=str(normalized_edge.get("reason", "") or ""),
                    origin="entity_graph",
                    bridges=bridges,
                )

    return relationships


def _build_community_index(
    file_sources: list[str],
    document_graph: dict[str, Any],
    entity_graph: dict[str, Any],
    semantic_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    file_sources = sorted({_normalize_source_path(source) for source in file_sources if _normalize_source_path(source)})
    relationships = _project_file_relationships(file_sources, document_graph, entity_graph)

    strong_adjacency: dict[str, set[str]] = {source: set() for source in file_sources}
    all_adjacency: dict[str, set[str]] = {source: set() for source in file_sources}
    for relation in relationships:
        source = relation["source"]
        target = relation["target"]
        all_adjacency.setdefault(source, set()).add(target)
        all_adjacency.setdefault(target, set()).add(source)
        if relation["kind"] in COMMUNITY_STRONG_EDGE_TYPES:
            strong_adjacency.setdefault(source, set()).add(target)
            strong_adjacency.setdefault(target, set()).add(source)

    communities_raw: list[list[str]] = []
    seen_sources: set[str] = set()
    for source in file_sources:
        if source in seen_sources:
            continue
        queue = deque([source])
        component: list[str] = []
        seen_sources.add(source)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(strong_adjacency.get(current, set())):
                if neighbor in seen_sources:
                    continue
                seen_sources.add(neighbor)
                queue.append(neighbor)
        communities_raw.append(sorted(component))

    communities_raw.sort(key=lambda items: (-len(items), items[0] if items else ""))
    file_to_community: dict[str, str] = {}
    community_ids: list[str] = []
    for index, files in enumerate(communities_raw, start=1):
        community_id = f"community-{index:03d}"
        community_ids.append(community_id)
        for source in files:
            file_to_community[source] = community_id

    entity_nodes = {
        str(node.get("id", "")): node
        for node in entity_graph.get("nodes", [])
        if isinstance(node, dict) and node.get("id")
    }
    entity_edges = [
        edge
        for edge in entity_graph.get("edges", [])
        if isinstance(edge, dict)
    ]
    semantic_edge_counts: dict[str, int] = {}
    normalized_entity_edges = [
        normalized_edge
        for edge in entity_edges
        if (normalized_edge := _normalize_entity_edge_payload(edge)) is not None
    ]
    semantic_attachment_files = _entity_attachment_files(
        entity_nodes,
        normalized_entity_edges,
    )
    attachment_node_ids = set(semantic_attachment_files.keys())
    for edge in normalized_entity_edges:
        source_id = str(edge.get("source", "") or "")
        target_id = str(edge.get("target", "") or "")
        if source_id in attachment_node_ids:
            semantic_edge_counts[source_id] = semantic_edge_counts.get(source_id, 0) + 1
        if target_id in attachment_node_ids:
            semantic_edge_counts[target_id] = semantic_edge_counts.get(target_id, 0) + 1

    communities: list[dict[str, Any]] = []
    for community_id, files in zip(community_ids, communities_raw):
        file_set = set(files)
        kbs = sorted({_extract_kb(source) for source in files if _extract_kb(source)})
        file_degrees = {
            source: len({neighbor for neighbor in all_adjacency.get(source, set()) if neighbor in file_set})
            for source in files
        }
        top_files = [
            {
                "source": source,
                "degree": file_degrees[source],
                "kb": _extract_kb(source),
            }
            for source in sorted(files, key=lambda item: (-file_degrees[item], item))
        ][:COMMUNITY_TOP_FILES]

        symbol_scores: dict[str, tuple[dict[str, Any], int]] = {}
        for node_id, node in entity_nodes.items():
            if node.get("type") != "symbol":
                continue
            if node.get("symbol_kind") in {"module", "import"}:
                continue
            source = _entity_node_file_source(node)
            if source not in file_set:
                continue
            score = 0
            for edge in entity_edges:
                if edge.get("source") == node_id or edge.get("target") == node_id:
                    score += 1
            symbol_scores[node_id] = (node, score)

        top_symbols = [
            {
                "id": str(node.get("id", "")),
                "name": str(node.get("qualified_name") or node.get("name") or ""),
                "source": _entity_node_file_source(node),
                "score": score,
            }
            for node, score in sorted(
                symbol_scores.values(),
                key=lambda item: (-item[1], str(item[0].get("qualified_name") or item[0].get("name") or "")),
            )
        ][:COMMUNITY_TOP_SYMBOLS]
        top_concepts = [
            {
                "id": node_id,
                "name": str(node.get("name") or ""),
                "file_count": len(attached_files & file_set),
                "score": len(attached_files & file_set) * 100 + semantic_edge_counts.get(node_id, 0),
            }
            for node_id, node in entity_nodes.items()
            if node.get("type") == "concept"
            and (attached_files := semantic_attachment_files.get(node_id, set()))
            and attached_files & file_set
        ]
        top_concepts = sorted(
            top_concepts,
            key=lambda item: (-int(item["file_count"]), -int(item["score"]), str(item["name"])),
        )[:COMMUNITY_TOP_SYMBOLS]
        top_decisions = [
            {
                "id": node_id,
                "name": str(node.get("name") or ""),
                "file_count": len(attached_files & file_set),
                "score": len(attached_files & file_set) * 100 + semantic_edge_counts.get(node_id, 0),
            }
            for node_id, node in entity_nodes.items()
            if node.get("type") == "decision"
            and (attached_files := semantic_attachment_files.get(node_id, set()))
            and attached_files & file_set
        ]
        top_decisions = sorted(
            top_decisions,
            key=lambda item: (-int(item["file_count"]), -int(item["score"]), str(item["name"])),
        )[:COMMUNITY_TOP_SYMBOLS]
        top_query_notes = [
            {
                "id": node_id,
                "name": str(node.get("name") or ""),
                "file_count": len(attached_files & file_set),
                "score": len(attached_files & file_set) * 100 + semantic_edge_counts.get(node_id, 0),
                "note_relpath": str(node.get("note_relpath") or ""),
            }
            for node_id, node in entity_nodes.items()
            if node.get("type") == "query_note"
            and (attached_files := semantic_attachment_files.get(node_id, set()))
            and attached_files & file_set
        ]
        top_query_notes = sorted(
            top_query_notes,
            key=lambda item: (-int(item["file_count"]), -int(item["score"]), str(item["name"])),
        )[:COMMUNITY_TOP_QUERY_NOTES]

        label = (
            top_symbols[0]["name"]
            if top_symbols
            else top_concepts[0]["name"]
            if top_concepts
            else top_query_notes[0]["name"]
            if top_query_notes
            else Path(top_files[0]["source"]).stem if top_files else community_id
        )
        suggested_queries: list[str] = []
        if top_symbols:
            suggested_queries.append(f"{top_symbols[0]['name']} 在哪里实现？")
        if top_concepts:
            suggested_queries.append(f"{top_concepts[0]['name']} 在哪些文件里被讨论？")
        if top_decisions:
            suggested_queries.append(f"为什么采用 {top_decisions[0]['name']}？")
        if top_query_notes:
            suggested_queries.append(f"已保存问题“{top_query_notes[0]['name']}”涉及哪些文件？")
        if top_files:
            suggested_queries.append(f"{Path(top_files[0]['source']).stem} 相关流程是什么？")
        if kbs:
            suggested_queries.append(f"{kbs[0]} 这个社区主要关注什么？")

        communities.append(
            {
                "id": community_id,
                "label": label,
                "size": len(files),
                "files": files,
                "kbs": kbs,
                "top_files": top_files,
                "top_symbols": top_symbols,
                "top_concepts": top_concepts,
                "top_decisions": top_decisions,
                "top_query_notes": top_query_notes,
                "suggested_queries": suggested_queries[:3],
            }
        )

    file_degree_overall = {
        source: len(neighbors)
        for source, neighbors in all_adjacency.items()
    }
    god_nodes: list[dict[str, Any]] = [
        {
            "id": _entity_file_node_id(source),
            "type": "file",
            "name": Path(source).name,
            "source": source,
            "degree": file_degree_overall.get(source, 0),
        }
        for source in sorted(file_sources, key=lambda item: (-file_degree_overall.get(item, 0), item))
    ][:COMMUNITY_TOP_GOD_NODES]

    bridges: list[dict[str, Any]] = []
    for relation in relationships:
        source = relation["source"]
        target = relation["target"]
        source_community = file_to_community.get(source)
        target_community = file_to_community.get(target)
        if not source_community or not target_community or source_community == target_community:
            continue
        bridges.append(
            {
                "source_community": source_community,
                "target_community": target_community,
                "source": source,
                "target": target,
                "kind": relation["kind"],
                "reason": relation["reason"],
                "origin": relation["origin"],
                "bridges": relation["bridges"],
            }
        )

    bridges.sort(
        key=lambda item: (
            0 if item["origin"] == "entity_graph" else 1,
            _entity_expansion_priority(item["kind"]),
            item["source_community"],
            item["target_community"],
            item["source"],
            item["target"],
        )
    )
    concept_count = sum(1 for node in entity_nodes.values() if node.get("type") == "concept")
    decision_count = sum(1 for node in entity_nodes.values() if node.get("type") == "decision")
    query_note_count = sum(1 for node in entity_nodes.values() if node.get("type") == "query_note")
    semantic_summary = {
        "enabled": bool((semantic_stats or {}).get("enabled")),
        "disabled_reason": str((semantic_stats or {}).get("disabled_reason", "") or ""),
        "concept_count": concept_count,
        "decision_count": decision_count,
        "query_note_count": query_note_count,
        "semantic_node_count": concept_count + decision_count,
        "semantic_edge_count": sum(
            1
            for edge in entity_edges
            if str(edge.get("type", "") or "") in {"semantically_related", "rationale_for"}
        ),
        "cached_sections": int((semantic_stats or {}).get("cached_sections") or 0),
        "extracted_sections": int((semantic_stats or {}).get("extracted_sections") or 0),
        "failed_sections": int((semantic_stats or {}).get("failed_sections") or 0),
        "api_calls": int((semantic_stats or {}).get("api_calls") or 0),
        "total_tokens": int((semantic_stats or {}).get("total_tokens") or 0),
        "duration_seconds": float((semantic_stats or {}).get("duration_seconds") or 0.0),
    }

    return {
        "version": 1,
        "file_count": len(file_sources),
        "relationship_count": len(relationships),
        "community_count": len(communities),
        "semantic_summary": semantic_summary,
        "communities": communities,
        "file_to_community": file_to_community,
        "god_nodes": god_nodes,
        "bridges": bridges[:COMMUNITY_TOP_BRIDGES],
    }


def _format_report_list(items: list[str], empty_text: str) -> list[str]:
    if not items:
        return [f"- {empty_text}"]
    return [f"- {item}" for item in items]


def _render_graph_report(
    community_index: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    build_time = str(manifest.get("build_time", "") or "unknown")
    source_dir = str(manifest.get("source_dir", "") or "unknown")
    file_count = int(community_index.get("file_count") or 0)
    relationship_count = int(community_index.get("relationship_count") or 0)
    community_count = int(community_index.get("community_count") or 0)
    communities = community_index.get("communities", [])
    god_nodes = community_index.get("god_nodes", [])
    bridges = community_index.get("bridges", [])
    semantic_summary = (
        community_index.get("semantic_summary")
        if isinstance(community_index.get("semantic_summary"), dict)
        else {}
    )

    semantic_lines = [
        "# 图谱报告",
        "",
        "此报告由构建阶段自动生成，当前用于帮助理解知识库结构。",
        "",
        f"- 最近构建：{build_time}",
        f"- 源目录：{source_dir}",
        f"- 文件总数：{file_count}",
        f"- 社区数量：{community_count}",
        f"- 关系总数：{relationship_count}",
        "",
        "## 语义抽取",
        "",
        f"- 启用状态：{'enabled' if semantic_summary.get('enabled') else 'disabled'}",
    ]
    if semantic_summary.get("disabled_reason"):
        semantic_lines.append(f"- 禁用原因：{semantic_summary.get('disabled_reason')}")
    semantic_lines.extend(
        [
        f"- 语义节点：{int(semantic_summary.get('semantic_node_count') or 0)}",
        f"- Concepts：{int(semantic_summary.get('concept_count') or 0)}",
        f"- Decisions：{int(semantic_summary.get('decision_count') or 0)}",
        f"- Query Notes：{int(semantic_summary.get('query_note_count') or 0)}",
        f"- 语义边：{int(semantic_summary.get('semantic_edge_count') or 0)}",
        f"- API 调用：{int(semantic_summary.get('api_calls') or 0)}",
        f"- 缓存命中 section：{int(semantic_summary.get('cached_sections') or 0)}",
        f"- 总 tokens：{int(semantic_summary.get('total_tokens') or 0)}",
        f"- 耗时：{float(semantic_summary.get('duration_seconds') or 0.0):.3f}s",
        "",
        "## God Nodes",
        "",
        ]
    )
    lines = semantic_lines

    god_node_lines = [
        f"{item.get('source', '')} (degree={item.get('degree', 0)})"
        for item in god_nodes
        if isinstance(item, dict) and item.get("source")
    ]
    lines.extend(_format_report_list(god_node_lines, "暂无显著 hub 节点"))
    lines.extend(["", "## 社区概览", ""])

    if not isinstance(communities, list) or not communities:
        lines.extend(["- 暂无社区数据", ""])
    else:
        for community in communities:
            if not isinstance(community, dict):
                continue
            community_id = str(community.get("id", "") or "community")
            label = str(community.get("label", "") or community_id)
            size = int(community.get("size") or 0)
            kbs = ", ".join(str(item) for item in community.get("kbs", []) if item) or "未分类"
            lines.extend(
                [
                    f"### {community_id}: {label}",
                    "",
                    f"- 文件数：{size}",
                    f"- 知识库：{kbs}",
                    "- Top Files:",
                ]
            )
            top_file_lines = [
                f"{item.get('source', '')} (degree={item.get('degree', 0)})"
                for item in community.get("top_files", [])
                if isinstance(item, dict) and item.get("source")
            ]
            lines.extend(_format_report_list(top_file_lines, "暂无文件摘要"))
            lines.append("- Top Symbols:")
            top_symbol_lines = [
                f"{item.get('name', '')} @ {item.get('source', '')} (score={item.get('score', 0)})"
                for item in community.get("top_symbols", [])
                if isinstance(item, dict) and item.get("name")
            ]
            lines.extend(_format_report_list(top_symbol_lines, "暂无符号摘要"))
            lines.append("- Top Concepts:")
            top_concept_lines = [
                f"{item.get('name', '')} (files={item.get('file_count', 0)}, score={item.get('score', 0)})"
                for item in community.get("top_concepts", [])
                if isinstance(item, dict) and item.get("name")
            ]
            lines.extend(_format_report_list(top_concept_lines, "暂无语义概念"))
            lines.append("- Top Decisions:")
            top_decision_lines = [
                f"{item.get('name', '')} (files={item.get('file_count', 0)}, score={item.get('score', 0)})"
                for item in community.get("top_decisions", [])
                if isinstance(item, dict) and item.get("name")
            ]
            lines.extend(_format_report_list(top_decision_lines, "暂无语义决策"))
            lines.append("- Top Query Notes:")
            top_query_note_lines = [
                f"{item.get('name', '')} (files={item.get('file_count', 0)}, path={item.get('note_relpath', '')})"
                for item in community.get("top_query_notes", [])
                if isinstance(item, dict) and item.get("name")
            ]
            lines.extend(_format_report_list(top_query_note_lines, "暂无知识笔记"))
            lines.append("- Suggested Questions:")
            suggested_query_lines = [
                str(item)
                for item in community.get("suggested_queries", [])
                if isinstance(item, str) and item
            ]
            lines.extend(_format_report_list(suggested_query_lines, "暂无建议问题"))
            lines.append("")

    lines.extend(["## 跨社区连接", ""])
    bridge_lines: list[str] = []
    if isinstance(bridges, list):
        for bridge in bridges:
            if not isinstance(bridge, dict):
                continue
            source_community = str(bridge.get("source_community", "") or "?")
            target_community = str(bridge.get("target_community", "") or "?")
            kind = str(bridge.get("kind", "") or "unknown")
            source = str(bridge.get("source", "") or "")
            target = str(bridge.get("target", "") or "")
            reason = str(bridge.get("reason", "") or "")
            origin = str(bridge.get("origin", "") or "")
            bridge_summary = ""
            raw_bridge_entities = bridge.get("bridges", [])
            if isinstance(raw_bridge_entities, list) and raw_bridge_entities:
                names = [
                    str(item.get("name", "") or item.get("id", ""))
                    for item in raw_bridge_entities
                    if isinstance(item, dict)
                ]
                bridge_summary = f" | bridges: {', '.join(name for name in names if name)}" if names else ""
            bridge_lines.append(
                f"{source_community} -> {target_community} | {kind} | {source} -> {target} | reason: {reason} | origin: {origin}{bridge_summary}"
            )
    lines.extend(_format_report_list(bridge_lines, "暂无跨社区连接"))
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# 向量库 / SearchBundle 操作
# ─────────────────────────────────────────────────────────
def _clear_generated_wiki_artifacts(persist_path: Path) -> None:
    wiki_dir = persist_path / WIKI_DIRNAME
    if not wiki_dir.exists():
        return
    for dirname in ("files", "communities", "entities"):
        generated_dir = wiki_dir / dirname
        if generated_dir.exists():
            shutil.rmtree(generated_dir)
    for filename in ("index.md", "log.md"):
        page_path = wiki_dir / filename
        if page_path.exists():
            page_path.unlink()


def _write_index_artifacts(
    persist_path: Path,
    indexed_files: list[IndexedFile],
    embed_model: str,
    source_dir: str,
    total_chunks: int,
    staged_normalized_dir: Path | None = None,
    llm_api_key: str = "",
    llm_model: str = "",
    llm_base_url: str = "",
) -> dict[str, Any]:
    normalized_dir = persist_path / NORMALIZED_TEXT_DIRNAME
    if normalized_dir.exists():
        shutil.rmtree(normalized_dir)

    if staged_normalized_dir is not None and staged_normalized_dir.exists():
        normalized_dir.parent.mkdir(parents=True, exist_ok=True)
        staged_normalized_dir.rename(normalized_dir)
        for indexed in indexed_files:
            if indexed.normalized_text_path is not None and indexed.normalized_text is None:
                indexed.normalized_text_path = normalized_dir / f"{indexed.rel_path}.txt"
    else:
        normalized_dir.mkdir(parents=True, exist_ok=True)

    skip_graph = _env_flag("SKIP_GRAPH")
    skip_semantic = skip_graph or _env_flag("SKIP_SEMANTIC")
    skip_wiki = _env_flag("SKIP_WIKI")
    manifest = {
        "build_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "embed_model": embed_model,
        "total_chunks": total_chunks,
        "kb_enabled": True,
        "source_dir": str(Path(source_dir).expanduser()),
        "normalized_text_dir": NORMALIZED_TEXT_DIRNAME,
        "symbol_index_file": SYMBOL_INDEX_FILENAME,
        "document_graph_file": DOCUMENT_GRAPH_FILENAME,
        "entity_graph_file": ENTITY_GRAPH_FILENAME,
        "semantic_extract_cache_file": SEMANTIC_EXTRACT_CACHE_FILENAME,
        "community_index_file": COMMUNITY_INDEX_FILENAME,
        "graph_report_file": f"{REPORTS_DIRNAME}/{GRAPH_REPORT_FILENAME}",
        "lint_report_file": LINT_REPORT_FILENAME,
        "search_mode_default": os.getenv("SEARCH_MODE", DEFAULT_SEARCH_MODE).strip() or DEFAULT_SEARCH_MODE,
        "semantic_graph_stats": {},
        "build_flags": {
            "skip_graph": skip_graph,
            "skip_semantic": skip_semantic,
            "skip_wiki": skip_wiki,
        },
        "files": [],
    }

    symbol_index_path = persist_path / SYMBOL_INDEX_FILENAME
    symbol_handle = None
    wrote_symbols = False
    for indexed in indexed_files:
        normalized_rel = f"{indexed.rel_path}.txt"
        cache_path = normalized_dir / normalized_rel
        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            indexed.write_normalized_text(cache_path)

        stat = indexed.file_path.stat()
        manifest["files"].append(
            {
                "name": indexed.rel_path,
                "kb": indexed.kb,
                "suffix": indexed.suffix,
                "size_kb": round(stat.st_size / 1024, 1),
                "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "chunks": indexed.chunk_count,
                "original_chunks": indexed.original_chunk_count,
                "truncated": indexed.truncated,
                "normalized_text": normalized_rel,
            }
        )
        for record in indexed.symbols:
            if symbol_handle is None:
                symbol_index_path.parent.mkdir(parents=True, exist_ok=True)
                symbol_handle = symbol_index_path.open("w", encoding="utf-8")
            symbol_handle.write(json.dumps(record, ensure_ascii=False))
            symbol_handle.write("\n")
            wrote_symbols = True

    if symbol_handle is not None:
        symbol_handle.close()
    if not wrote_symbols and symbol_index_path.exists():
        symbol_index_path.unlink()

    file_sources = [_normalize_source_path(indexed.rel_path) for indexed in indexed_files]
    if skip_graph:
        document_graph = {"version": 1, "edge_count": 0, "neighbors": {}}
    else:
        document_graph = _build_document_graph(indexed_files)
    document_graph_path = persist_path / DOCUMENT_GRAPH_FILENAME
    document_graph_path.write_text(
        json.dumps(document_graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    semantic_sections: list[dict[str, Any]] = []
    if skip_semantic:
        _write_semantic_cache(persist_path, {"version": 1, "entries": {}})
        semantic_stats = {
            "enabled": False,
            "reason": "skipped_by_graph" if skip_graph else "skipped_by_env",
            "api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "concept_count": 0,
            "decision_count": 0,
        }
    else:
        semantic_sections, semantic_stats = _extract_semantic_sections(
            indexed_files=indexed_files,
            persist_path=persist_path,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
        )
    manifest["semantic_graph_stats"] = semantic_stats

    if skip_graph:
        entity_graph = _build_file_only_entity_graph(file_sources, document_graph)
    else:
        entity_graph = _build_entity_graph(
            indexed_files,
            document_graph=document_graph,
            semantic_sections=semantic_sections,
        )
    entity_graph = _merge_query_notes_into_entity_graph(
        entity_graph,
        _load_query_note_records(persist_path),
    )
    entity_graph_path = persist_path / ENTITY_GRAPH_FILENAME
    entity_graph_path.write_text(
        json.dumps(entity_graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    community_index = _build_community_index(
        file_sources=file_sources,
        document_graph=document_graph,
        entity_graph=entity_graph,
        semantic_stats=semantic_stats,
    )
    community_index_path = persist_path / COMMUNITY_INDEX_FILENAME
    community_index_path.write_text(
        json.dumps(community_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    reports_dir = persist_path / REPORTS_DIRNAME
    reports_dir.mkdir(parents=True, exist_ok=True)
    graph_report_path = reports_dir / GRAPH_REPORT_FILENAME
    graph_report_path.write_text(
        _render_graph_report(community_index, manifest),
        encoding="utf-8",
    )

    (persist_path / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def refresh_query_note_graph_artifacts(persist_path: Path) -> None:
    manifest_path = persist_path / "index_manifest.json"
    document_graph_path = persist_path / DOCUMENT_GRAPH_FILENAME
    entity_graph_path = persist_path / ENTITY_GRAPH_FILENAME
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(manifest, dict):
        return

    document_graph: dict[str, Any] = {"version": 1, "edge_count": 0, "neighbors": {}}
    if document_graph_path.exists():
        try:
            loaded_document_graph = json.loads(document_graph_path.read_text(encoding="utf-8"))
            if isinstance(loaded_document_graph, dict):
                document_graph = loaded_document_graph
        except (OSError, json.JSONDecodeError):
            document_graph = {"version": 1, "edge_count": 0, "neighbors": {}}

    file_sources = [
        _normalize_source_path(str(entry.get("name", "") or ""))
        for entry in manifest.get("files", [])
        if isinstance(entry, dict)
    ]

    entity_graph: dict[str, Any]
    if entity_graph_path.exists():
        try:
            loaded_entity_graph = json.loads(entity_graph_path.read_text(encoding="utf-8"))
            entity_graph = (
                loaded_entity_graph
                if isinstance(loaded_entity_graph, dict)
                else _build_file_only_entity_graph(file_sources, document_graph)
            )
        except (OSError, json.JSONDecodeError):
            entity_graph = _build_file_only_entity_graph(file_sources, document_graph)
    else:
        entity_graph = _build_file_only_entity_graph(file_sources, document_graph)

    entity_graph = _merge_query_notes_into_entity_graph(
        entity_graph,
        _load_query_note_records(persist_path),
    )
    entity_graph_path.write_text(
        json.dumps(entity_graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    community_index = _build_community_index(
        file_sources=file_sources,
        document_graph=document_graph,
        entity_graph=entity_graph,
        semantic_stats=manifest.get("semantic_graph_stats", {}),
    )
    community_index_path = persist_path / COMMUNITY_INDEX_FILENAME
    community_index_path.write_text(
        json.dumps(community_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    reports_dir = persist_path / REPORTS_DIRNAME
    reports_dir.mkdir(parents=True, exist_ok=True)
    graph_report_path = reports_dir / GRAPH_REPORT_FILENAME
    graph_report_path.write_text(
        _render_graph_report(community_index, manifest),
        encoding="utf-8",
    )


def _iter_document_batches(documents: Iterable[Document], batch_size: int) -> Iterable[list[Document]]:
    batch: list[Document] = []
    for document in documents:
        batch.append(document)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _document_batch_ids(batch: list[Document]) -> list[str] | None:
    ids = [getattr(document, "id", None) for document in batch]
    if any(ids):
        return [str(doc_id or "") for doc_id in ids]
    return None


def _embed_texts_with_retries(
    *,
    texts: list[str],
    embeddings_factory: Callable[[], OpenAIEmbeddings],
    max_retries: int,
    retry_base_seconds: float,
    rate_limit_callback: Callable[[float, int, int], None] | None = None,
) -> list[list[float]]:
    for attempt in range(max_retries + 1):
        try:
            return embeddings_factory().embed_documents(texts)
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt >= max_retries:
                raise
            delay = retry_base_seconds * (2**attempt)
            if rate_limit_callback is not None:
                rate_limit_callback(delay, attempt + 1, max_retries)
            time.sleep(delay)
    raise RuntimeError("embedding 批处理在重试后仍未成功。")


def _add_embedded_batch(
    *,
    batch: list[Document],
    embedded_texts: list[list[float]],
    embeddings: OpenAIEmbeddings,
    vectorstore: FAISS | None,
) -> FAISS:
    texts = [document.page_content for document in batch]
    metadatas = [document.metadata for document in batch]
    ids = _document_batch_ids(batch)
    text_embeddings = zip(texts, embedded_texts)
    if vectorstore is None:
        # 兼容测试 / 外部插件把模块级 `FAISS` 替换成极简假对象的场景。
        if not hasattr(FAISS, "from_embeddings"):
            return FAISS.from_documents(batch, embeddings)
        return FAISS.from_embeddings(
            text_embeddings,
            embeddings,
            metadatas=metadatas,
            ids=ids,
        )
    # 同上：若外部替换了 `FAISS` 实现，只退回到旧的 document-based 接口。
    if not hasattr(vectorstore, "add_embeddings"):
        vectorstore.add_documents(batch)
        return vectorstore
    vectorstore.add_embeddings(
        text_embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    return vectorstore


def _build_vectorstore_from_document_stream(
    *,
    indexed_files: list[IndexedFile],
    documents: Iterable[Document],
    total_chunks: int,
    md_dir: str,
    embed_api_key: str,
    embed_base_url: str,
    embed_model: str,
    persist_dir: str,
    staged_normalized_dir: Path | None,
    progress_callback: Callable[[int, int, str], None] | None,
    llm_api_key: str,
    llm_model: str,
    llm_base_url: str,
) -> SearchBundle:
    def _cb(current: int, total: int, message: str) -> None:
        if progress_callback:
            progress_callback(current, total, message)

    if total_chunks == 0:
        raise ValueError(
            f"在 {md_dir} 中未找到可索引文本。支持后缀: {', '.join(sorted(SUPPORTED_TEXT_SUFFIXES))}"
        )

    _cb(0, total_chunks + 4, f"解析完毕，共 {len(indexed_files)} 个文件，{total_chunks} 个分片。")
    truncated_files = [indexed for indexed in indexed_files if indexed.truncated]
    if truncated_files:
        preview = "；".join(
            f"{indexed.rel_path}: {indexed.original_chunk_count} -> {indexed.chunk_count}"
            for indexed in truncated_files[:3]
        )
        _cb(
            0,
            total_chunks + 4,
            f"注意：{len(truncated_files)} 个文件因 MAX_CHUNKS_PER_FILE 被截断。{preview}",
        )
    embeddings = make_embeddings(api_key=embed_api_key, base_url=embed_base_url, model=embed_model)

    batch_size = max(1, int(os.getenv("EMBED_BATCH_SIZE", str(DEFAULT_EMBED_BATCH_SIZE))))
    # 默认不做固定节流；仅在真正触发限流时才指数退避，避免大语料重建被 sleep 吞掉数小时。
    batch_sleep_seconds = max(
        0.0,
        float(os.getenv("EMBED_BATCH_SLEEP_SECONDS", str(DEFAULT_EMBED_BATCH_SLEEP_SECONDS))),
    )
    embed_concurrency = max(1, int(os.getenv("EMBED_CONCURRENCY", "1")))
    max_retries = max(0, int(os.getenv("EMBED_MAX_RETRIES", "8")))
    retry_base_seconds = max(0.5, float(os.getenv("EMBED_RETRY_BASE_SECONDS", "5")))

    vectorstore: FAISS | None = None
    total_steps = total_chunks + 4
    processed_chunks = 0
    rate_limit_events: Queue[str] = Queue()

    def _emit_rate_limit_event(delay: float, attempt: int, retry_limit: int) -> None:
        message = f"触发 embedding 限流，{delay:.1f}s 后重试第 {attempt}/{retry_limit} 次..."
        if embed_concurrency <= 1:
            _cb(processed_chunks, total_steps, message)
            return
        rate_limit_events.put(message)

    def _drain_rate_limit_events() -> None:
        while True:
            try:
                message = rate_limit_events.get_nowait()
            except Empty:
                break
            _cb(processed_chunks, total_steps, message)

    if embed_concurrency <= 1:
        for batch in _iter_document_batches(documents, batch_size):
            embedded_texts = _embed_texts_with_retries(
                texts=[document.page_content for document in batch],
                embeddings_factory=lambda: embeddings,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                rate_limit_callback=_emit_rate_limit_event,
            )
            vectorstore = _add_embedded_batch(
                batch=batch,
                embedded_texts=embedded_texts,
                embeddings=embeddings,
                vectorstore=vectorstore,
            )
            processed_chunks += len(batch)
            _cb(processed_chunks, total_steps, f"向量化中 {processed_chunks}/{total_chunks}...")
            if batch_sleep_seconds > 0 and processed_chunks < total_chunks:
                time.sleep(batch_sleep_seconds)
    else:
        worker_state = local()
        if batch_sleep_seconds > 0:
            _cb(
                processed_chunks,
                total_steps,
                "EMBED_BATCH_SLEEP_SECONDS 仅在串行 embedding 下生效；并发模式下已忽略。",
            )

        def _worker_embeddings() -> OpenAIEmbeddings:
            client = getattr(worker_state, "client", None)
            if client is None:
                client = make_embeddings(
                    api_key=embed_api_key,
                    base_url=embed_base_url,
                    model=embed_model,
                )
                worker_state.client = client
            return client

        def _submit_embed_batch(
            executor: ThreadPoolExecutor,
            pending: dict[int, tuple[list[Document], Future[list[list[float]]]]],
            batch_iter: Iterable[list[Document]],
            seq: int,
        ) -> tuple[bool, int]:
            try:
                batch = next(batch_iter)
            except StopIteration:
                return False, seq
            pending[seq] = (
                batch,
                executor.submit(
                    _embed_texts_with_retries,
                    texts=[document.page_content for document in batch],
                    embeddings_factory=_worker_embeddings,
                    max_retries=max_retries,
                    retry_base_seconds=retry_base_seconds,
                    rate_limit_callback=_emit_rate_limit_event,
                ),
            )
            return True, seq + 1

        batch_iter = iter(_iter_document_batches(documents, batch_size))
        pending: dict[int, tuple[list[Document], Future[list[list[float]]]]] = {}
        next_seq = 0
        expected_seq = 0
        with ThreadPoolExecutor(max_workers=embed_concurrency) as executor:
            while len(pending) < embed_concurrency:
                submitted, next_seq = _submit_embed_batch(executor, pending, batch_iter, next_seq)
                if not submitted:
                    break

            while pending:
                batch, future = pending.pop(expected_seq)
                while True:
                    try:
                        embedded_texts = future.result(timeout=0.1)
                        break
                    except FutureTimeoutError:
                        _drain_rate_limit_events()
                _drain_rate_limit_events()
                vectorstore = _add_embedded_batch(
                    batch=batch,
                    embedded_texts=embedded_texts,
                    embeddings=embeddings,
                    vectorstore=vectorstore,
                )
                processed_chunks += len(batch)
                _cb(processed_chunks, total_steps, f"向量化中 {processed_chunks}/{total_chunks}...")
                expected_seq += 1

                while len(pending) < embed_concurrency:
                    submitted, next_seq = _submit_embed_batch(executor, pending, batch_iter, next_seq)
                    if not submitted:
                        break

    if vectorstore is None:
        raise RuntimeError("未生成任何向量分片。")

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(persist_path))
    _cb(total_chunks + 1, total_steps, "正在写入检索辅助索引...")

    manifest = _write_index_artifacts(
        persist_path=persist_path,
        indexed_files=indexed_files,
        embed_model=embed_model,
        source_dir=md_dir,
        total_chunks=total_chunks,
        staged_normalized_dir=staged_normalized_dir,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
    )
    if manifest.get("build_flags", {}).get("skip_wiki"):
        _clear_generated_wiki_artifacts(persist_path)
        _cb(total_chunks + 2, total_steps, "已按配置跳过离线 wiki 生成。")
    else:
        _cb(total_chunks + 2, total_steps, "正在生成离线 wiki 导航...")
        from wiki import generate_wiki

        generate_wiki(persist_path=persist_path, manifest=manifest)
    _read_cached_text.cache_clear()
    _cb(total_chunks + 3, total_steps, "正在加载检索 bundle...")

    bundle = load_search_bundle(
        embed_api_key=embed_api_key,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        persist_dir=persist_dir,
    )
    if bundle is None:
        raise RuntimeError("索引文件写入成功，但 SearchBundle 回读失败。")

    _cb(total_chunks + 4, total_steps, f"✅ 索引构建完成，已保存到 {persist_dir}")
    bundle.manifest = manifest
    return bundle


def build_vectorstore(
    md_dir: str,
    embed_api_key: str,
    embed_base_url: str = SILICONFLOW_BASE_URL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    persist_dir: str = DEFAULT_FAISS_DIR,
    progress_callback: Callable[[int, int, str], None] | None = None,
    llm_api_key: str = "",
    llm_model: str = "",
    llm_base_url: str = "",
) -> SearchBundle:
    """
    构建（或重建）FAISS 向量索引并持久化到本地。
    """

    def _cb(current: int, total: int, message: str) -> None:
        if progress_callback:
            progress_callback(current, total, message)

    _cb(0, 1, "正在扫描目录并解析文本文件...")
    # 仅兼容外部测试 / 插件 monkeypatch `_build_documents` 的旧入口。
    if _build_documents is not _ORIGINAL_BUILD_DOCUMENTS:
        documents, indexed_files = _build_documents(md_dir)
        return _build_vectorstore_from_document_stream(
            indexed_files=indexed_files,
            documents=documents,
            total_chunks=len(documents),
            md_dir=md_dir,
            embed_api_key=embed_api_key,
            embed_base_url=embed_base_url,
            embed_model=embed_model,
            persist_dir=persist_dir,
            staged_normalized_dir=None,
            progress_callback=progress_callback,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
        )

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".opencortex-build-", dir=str(persist_path)) as tmpdir:
        cache_root = Path(tmpdir) / NORMALIZED_TEXT_DIRNAME
        indexed_files, total_chunks = _index_source_files(md_dir, cache_root)
        return _build_vectorstore_from_document_stream(
            indexed_files=indexed_files,
            documents=_iter_documents_for_indexed_files(indexed_files),
            total_chunks=total_chunks,
            md_dir=md_dir,
            embed_api_key=embed_api_key,
            embed_base_url=embed_base_url,
            embed_model=embed_model,
            persist_dir=persist_dir,
            staged_normalized_dir=cache_root,
            progress_callback=progress_callback,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
        )


def load_search_bundle(
    embed_api_key: str,
    embed_base_url: str = SILICONFLOW_BASE_URL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    persist_dir: str = DEFAULT_FAISS_DIR,
) -> SearchBundle | None:
    index_file = Path(persist_dir) / "index.faiss"
    if not index_file.exists():
        return None

    embeddings = make_embeddings(api_key=embed_api_key, base_url=embed_base_url, model=embed_model)
    vectorstore = FAISS.load_local(
        str(persist_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    manifest_path = Path(persist_dir) / "index_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"files": []}
    )
    raw_files = manifest.get("files", [])
    files: list[dict[str, Any]] = []
    for entry in raw_files if isinstance(raw_files, list) else []:
        if not isinstance(entry, dict):
            continue
        normalized_entry = dict(entry)
        source = _normalize_source_path(normalized_entry.get("name", ""))
        if not source:
            continue
        normalized_entry["name"] = source
        normalized_entry["kb"] = _extract_kb(source) or normalized_entry.get("kb", "")
        normalized_text = normalized_entry.get("normalized_text", "")
        if isinstance(normalized_text, str):
            normalized_entry["normalized_text"] = _normalize_source_path(normalized_text)
        files.append(normalized_entry)
    manifest["files"] = files
    files_by_source = {entry.get("name", ""): entry for entry in files if entry.get("name")}
    normalized_text_dir = Path(persist_dir) / manifest.get("normalized_text_dir", NORMALIZED_TEXT_DIRNAME)
    symbol_index_path = Path(persist_dir) / manifest.get("symbol_index_file", SYMBOL_INDEX_FILENAME)
    document_graph_path = Path(persist_dir) / manifest.get("document_graph_file", DOCUMENT_GRAPH_FILENAME)
    entity_graph_path = Path(persist_dir) / manifest.get("entity_graph_file", ENTITY_GRAPH_FILENAME)

    symbol_index: list[dict[str, Any]] = []
    if symbol_index_path.exists():
        for line in symbol_index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            source = _normalize_source_path(record.get("source", ""))
            if not source:
                continue
            record["source"] = source
            symbol_index.append(record)

    document_graph: dict[str, Any] = {"version": 1, "edge_count": 0, "neighbors": {}}
    if document_graph_path.exists():
        try:
            document_graph = json.loads(document_graph_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            document_graph = {"version": 1, "edge_count": 0, "neighbors": {}}
    graph_neighbors: dict[str, list[dict[str, Any]]] = {}
    raw_neighbors = document_graph.get("neighbors", {})
    if isinstance(raw_neighbors, dict):
        for source, edges in raw_neighbors.items():
            normalized_source = _normalize_source_path(source)
            if not normalized_source or not isinstance(edges, list):
                continue
            bucket = graph_neighbors.setdefault(normalized_source, [])
            seen_edges: set[tuple[str, str, str, int]] = set()
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                target = _normalize_source_path(edge.get("target", ""))
                if not target or target == normalized_source:
                    continue
                payload = dict(edge)
                payload["target"] = target
                key = (
                    str(payload.get("kind", "")),
                    target,
                    str(payload.get("reason", "")),
                    int(payload.get("hop") or 0),
                )
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                bucket.append(payload)
    document_graph = {
        **document_graph,
        "neighbors": graph_neighbors,
        "edge_count": sum(len(edges) for edges in graph_neighbors.values()),
    }

    raw_entity_graph: dict[str, Any] = {"version": 1, "node_count": 0, "edge_count": 0, "nodes": [], "edges": []}
    if entity_graph_path.exists():
        try:
            raw_entity_graph = json.loads(entity_graph_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw_entity_graph = {"version": 1, "node_count": 0, "edge_count": 0, "nodes": [], "edges": []}

    entity_nodes: list[dict[str, Any]] = []
    entity_nodes_by_id: dict[str, dict[str, Any]] = {}
    raw_nodes = raw_entity_graph.get("nodes", [])
    if isinstance(raw_nodes, list):
        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict):
                continue
            normalized_node = _normalize_entity_node_payload(raw_node)
            if normalized_node is None:
                continue
            node_id = str(normalized_node["id"])
            if node_id in entity_nodes_by_id:
                continue
            entity_nodes_by_id[node_id] = normalized_node
            entity_nodes.append(normalized_node)

    entity_edges: list[dict[str, Any]] = []
    entity_edges_by_source: dict[str, list[dict[str, Any]]] = {}
    seen_entity_edges: set[tuple[Any, ...]] = set()
    raw_edges = raw_entity_graph.get("edges", [])
    if isinstance(raw_edges, list):
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                continue
            normalized_edge = _normalize_entity_edge_payload(raw_edge)
            if normalized_edge is None:
                continue
            evidence = normalized_edge.get("evidence")
            evidence_source = ""
            evidence_line_start = None
            evidence_line_end = None
            if isinstance(evidence, dict):
                evidence_source = str(evidence.get("source", "") or "")
                evidence_line_start = evidence.get("line_start")
                evidence_line_end = evidence.get("line_end")
            dedupe_key = (
                normalized_edge["source"],
                normalized_edge["target"],
                normalized_edge.get("type", ""),
                normalized_edge.get("reason", ""),
                evidence_source,
                evidence_line_start,
                evidence_line_end,
            )
            if dedupe_key in seen_entity_edges:
                continue
            seen_entity_edges.add(dedupe_key)
            entity_edges.append(normalized_edge)
            entity_edges_by_source.setdefault(str(normalized_edge["source"]), []).append(normalized_edge)

    entity_graph = {
        **raw_entity_graph,
        "nodes": entity_nodes,
        "edges": entity_edges,
        "node_count": len(entity_nodes),
        "edge_count": len(entity_edges),
    }

    source_dir = manifest.get("source_dir")
    wiki_pages = _load_wiki_pages(Path(persist_dir), files)
    return SearchBundle(
        vectorstore=vectorstore,
        persist_dir=Path(persist_dir),
        source_dir=Path(source_dir) if source_dir else None,
        manifest=manifest,
        files=files,
        files_by_source=files_by_source,
        normalized_text_dir=normalized_text_dir,
        symbol_index=symbol_index,
        document_graph=document_graph,
        graph_neighbors=graph_neighbors,
        entity_graph=entity_graph,
        entity_nodes_by_id=entity_nodes_by_id,
        entity_edges_by_source=entity_edges_by_source,
        wiki_pages=wiki_pages,
    )


def load_vectorstore(
    embed_api_key: str,
    embed_base_url: str = SILICONFLOW_BASE_URL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    persist_dir: str = DEFAULT_FAISS_DIR,
) -> FAISS | None:
    bundle = load_search_bundle(
        embed_api_key=embed_api_key,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        persist_dir=persist_dir,
    )
    return bundle.vectorstore if bundle else None


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
# 检索：query plan / helper
# ─────────────────────────────────────────────────────────
@lru_cache(maxsize=2048)
def _read_cached_text(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8", errors="ignore")


def _search_mode(value: str | None) -> str:
    mode = (value or os.getenv("SEARCH_MODE", DEFAULT_SEARCH_MODE)).strip().lower()
    return mode if mode in SEARCH_MODES else DEFAULT_SEARCH_MODE


def _search_max_steps() -> int:
    raw = os.getenv("SEARCH_MAX_STEPS", str(SEARCH_MAX_STEPS_DEFAULT)).strip()
    try:
        value = int(raw)
    except ValueError:
        return SEARCH_MAX_STEPS_DEFAULT
    return 1 if value <= 1 else 2


def _bundle_sources(
    bundle: SearchBundle,
    kb: str | None = None,
    allowed_sources: set[str] | None = None,
) -> list[str]:
    results: list[str] = []
    for entry in bundle.files:
        source = entry.get("name", "")
        if not source:
            continue
        if kb is not None and entry.get("kb") != kb:
            continue
        if allowed_sources is not None and source not in allowed_sources:
            continue
        results.append(source)
    return results


def _has_supported_file_suffix(token: str) -> bool:
    clean = str(token).strip().lower()
    return any(clean.endswith(suffix) and len(clean) > len(suffix) for suffix in SUPPORTED_TEXT_SUFFIXES)


def _looks_like_explicit_source_term(token: str) -> bool:
    clean = str(token).strip()
    if not clean:
        return False
    if "*" in clean or "/" in clean or "\\" in clean:
        return True
    if clean.startswith(".") and clean.count(".") == 1 and len(clean) <= 8:
        return True
    return _has_supported_file_suffix(clean)


def _is_extension_only_glob(pattern: str) -> bool:
    stripped = str(pattern).strip().strip("*")
    return stripped.startswith(".") and stripped.count(".") == 1 and "/" not in stripped and "\\" not in stripped


def _extract_query_plan(question: str) -> QueryPlan:
    quoted = re.findall(r"`([^`]+)`", question)
    quoted += re.findall(r'"([^"]+)"', question)
    quoted += re.findall(r"“([^”]+)”", question)
    quoted += re.findall(r"'([^']+)'", question)
    quoted_clean_map: dict[str, str] = {}
    for item in quoted:
        clean = item.strip().strip(".,:;!?()[]{}")
        if clean:
            quoted_clean_map[clean.lower()] = clean

    raw_tokens = quoted + TOKEN_RE.findall(question) + NON_ASCII_FILENAME_RE.findall(question)
    path_globs: list[str] = []
    symbols: list[str] = []
    keywords: list[str] = []

    for token in raw_tokens:
        clean = token.strip().strip(".,:;!?()[]{}")
        if not clean:
            continue
        has_supported_suffix = _has_supported_file_suffix(clean)
        if clean.lower() not in _EXTENSION_ALIASES:
            keywords.append(clean)

        if _looks_like_explicit_source_term(clean):
            if clean.startswith(".") and clean.count(".") == 1 and "/" not in clean:
                path_globs.append(f"*{clean}")
            elif "*" in clean:
                path_globs.append(clean)
            else:
                path_globs.append(f"*{clean}*")

        if clean.startswith(".") and clean.count(".") == 1 and len(clean) <= 8:
            path_globs.append(f"*{clean}")

        stripped_call = clean[:-2] if clean.endswith("()") else clean
        quoted_token = stripped_call.lower() in quoted_clean_map or clean.lower() in quoted_clean_map
        if SYMBOL_RE.fullmatch(stripped_call) and (
            quoted_token or "_" in stripped_call or any(char.isupper() for char in stripped_call)
        ):
            symbols.append(stripped_call)

        if quoted_token and "." in clean and all(SYMBOL_RE.fullmatch(part) for part in clean.split(".")):
            symbols.append(clean)

        if not has_supported_suffix and "." in stripped_call and all(part for part in stripped_call.split(".")):
            tail = stripped_call.split(".")[-1]
            if SYMBOL_RE.fullmatch(tail):
                symbols.append(tail)

    for match in CALL_RE.findall(question):
        symbols.append(match)
        keywords.append(f"{match}(")

    lowered_question = question.lower()
    for alias, pattern in _EXTENSION_ALIASES.items():
        if re.search(rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])", lowered_question):
            path_globs.append(pattern)

    return QueryPlan(
        symbols=_dedupe_strings(symbols),
        keywords=_dedupe_strings(keywords),
        path_globs=_dedupe_strings(path_globs),
        semantic_query=question.strip(),
        reason="rule planner",
    )


def _hit_priority(match_kind: str) -> int:
    order = {"ast": 4, "grep": 3, "vector": 2, "glob": 1}
    return order.get(match_kind, 0)


def _merge_primary_kind(current: SearchHit, candidate: SearchHit) -> None:
    if _hit_priority(candidate.match_kind) > _hit_priority(current.match_kind):
        current.match_kind = candidate.match_kind
        current.metadata.update(candidate.metadata)
        if candidate.snippet:
            current.snippet = candidate.snippet
        if candidate.line_start is not None:
            current.line_start = candidate.line_start
        if candidate.line_end is not None:
            current.line_end = candidate.line_end


def _finalize_hits(
    grouped_hits: dict[str, list[SearchHit]],
    top_k: int,
) -> list[SearchHit]:
    merged: dict[tuple[str, int | None, int | None, str], SearchHit] = {}
    for match_kind, hits in grouped_hits.items():
        weight = RRF_WEIGHTS[match_kind]
        for rank, hit in enumerate(hits, start=1):
            key = hit.dedupe_key()
            fused = merged.get(key)
            if fused is None:
                fused = SearchHit(
                    source=hit.source,
                    match_kind=hit.match_kind,
                    snippet=hit.snippet,
                    score=0.0,
                    line_start=hit.line_start,
                    line_end=hit.line_end,
                    metadata=dict(hit.metadata),
                )
                merged[key] = fused

            fused.score += weight / (RRF_K + rank)
            if hit.metadata.get("exact_symbol"):
                fused.score += 0.25
            if hit.metadata.get("exact_path"):
                fused.score += 0.15
            _merge_primary_kind(fused, hit)

    ranked = sorted(
        merged.values(),
        key=lambda item: (item.score, _hit_priority(item.match_kind), item.source),
        reverse=True,
    )

    final_hits: list[SearchHit] = []
    per_file_count: dict[str, int] = {}
    for hit in ranked:
        if per_file_count.get(hit.source, 0) >= 2:
            continue
        final_hits.append(hit)
        per_file_count[hit.source] = per_file_count.get(hit.source, 0) + 1
        if len(final_hits) >= top_k:
            break
    return final_hits


def _first_non_empty_lines(cache_path: Path | None, limit: int = 3) -> str:
    if cache_path is None or not cache_path.exists():
        return ""
    lines = [line.strip() for line in _read_cached_text(str(cache_path)).splitlines() if line.strip()]
    return "\n".join(lines[:limit])


def glob_search(
    bundle: SearchBundle,
    path_globs: list[str],
    kb: str | None = None,
    allowed_sources: set[str] | None = None,
) -> list[SearchHit]:
    if not path_globs:
        return []

    results: list[SearchHit] = []
    for source in _bundle_sources(bundle, kb=kb, allowed_sources=allowed_sources):
        path_lower = source.lower()
        basename_lower = Path(source).name.lower()
        matched = False
        exact_path = False
        for pattern in path_globs:
            normalized_pattern = pattern.lower()
            if fnmatch.fnmatch(path_lower, normalized_pattern) or fnmatch.fnmatch(basename_lower, normalized_pattern):
                matched = True
                stripped = normalized_pattern.strip("*")
                exact_path = stripped == path_lower or stripped == basename_lower
                break
            stripped = normalized_pattern.strip("*")
            if stripped and (stripped in path_lower or stripped in basename_lower):
                matched = True
                exact_path = stripped == basename_lower or stripped == path_lower
                break
        if not matched:
            continue

        cache_path = bundle.cache_path_for(source)
        snippet = _first_non_empty_lines(cache_path) or source
        results.append(
            SearchHit(
                source=source,
                match_kind="glob",
                snippet=snippet,
                score=1.0,
                metadata={"exact_path": exact_path},
            )
        )

    return results


def _keyword_score(line: str, keyword: str) -> float:
    lowered_line = line.lower()
    lowered_keyword = keyword.lower()
    if lowered_keyword not in lowered_line:
        return 0.0
    if line == keyword:
        return 1.2
    if WORD_CHARS_RE.fullmatch(keyword):
        if re.search(rf"\b{re.escape(keyword)}\b", line, re.IGNORECASE):
            return 1.0
    return 0.75


def _grep_hits_from_matches(
    bundle: SearchBundle,
    matches: dict[tuple[str, int], dict[str, Any]],
) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for (source, line_no), payload in matches.items():
        cache_path = bundle.cache_path_for(source)
        if cache_path is None or not cache_path.exists():
            continue
        lines = _read_cached_text(str(cache_path)).splitlines()
        if not lines:
            continue
        start = max(1, line_no - 4)
        end = min(len(lines), line_no + 4)
        snippet = "\n".join(lines[start - 1 : end]).strip()
        exact_text = any(score >= 1.0 for score in payload["scores"])
        hits.append(
            SearchHit(
                source=source,
                match_kind="grep",
                snippet=snippet,
                score=max(payload["scores"]) + len(payload["keywords"]) * 0.05,
                line_start=line_no,
                line_end=line_no,
                metadata={
                    "matched_keywords": sorted(payload["keywords"]),
                    "exact_symbol": exact_text,
                },
            )
        )
    return sorted(hits, key=lambda item: (item.score, -int(item.line_start or 0)), reverse=True)


def _grep_search_with_rg(
    bundle: SearchBundle,
    keywords: list[str],
    sources: list[str],
) -> list[SearchHit]:
    rg_path = shutil.which("rg")
    if rg_path is None or not keywords or not sources:
        return []

    source_to_cache: dict[str, str] = {}
    for source in sources:
        cache_path = bundle.cache_path_for(source)
        if cache_path is None or not cache_path.exists():
            continue
        source_to_cache[source] = str(cache_path)
    if not source_to_cache:
        return []

    cache_to_source = {cache: source for source, cache in source_to_cache.items()}
    matches: dict[tuple[str, int], dict[str, Any]] = {}
    cmd = [
        rg_path,
        "-n",
        "-i",
        "--fixed-strings",
        "--no-heading",
        "--color=never",
    ]
    for keyword in keywords:
        cmd.extend(["-e", keyword])
    cmd.extend(source_to_cache.values())

    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    if completed.returncode not in (0, 1):
        return []

    for line in completed.stdout.splitlines():
        try:
            path_str, line_number_str, content = line.split(":", 2)
            source = cache_to_source.get(path_str)
            if source is None:
                continue
            line_number = int(line_number_str)
        except ValueError:
            continue

        matched_keywords: list[str] = []
        scores: list[float] = []
        for keyword in keywords:
            score = _keyword_score(content, keyword)
            if score <= 0:
                continue
            matched_keywords.append(keyword)
            scores.append(score)
        if not scores:
            continue

        key = (source, line_number)
        bucket = matches.setdefault(key, {"keywords": set(), "scores": []})
        bucket["keywords"].update(matched_keywords)
        bucket["scores"].extend(scores)

    return _grep_hits_from_matches(bundle, matches)


def _grep_search_python(
    bundle: SearchBundle,
    keywords: list[str],
    sources: list[str],
) -> list[SearchHit]:
    matches: dict[tuple[str, int], dict[str, Any]] = {}
    for source in sources:
        cache_path = bundle.cache_path_for(source)
        if cache_path is None or not cache_path.exists():
            continue
        lines = _read_cached_text(str(cache_path)).splitlines()
        for line_no, line in enumerate(lines, start=1):
            scores = []
            matched_keywords = []
            for keyword in keywords:
                score = _keyword_score(line, keyword)
                if score > 0:
                    scores.append(score)
                    matched_keywords.append(keyword)
            if not scores:
                continue
            key = (source, line_no)
            bucket = matches.setdefault(key, {"keywords": set(), "scores": []})
            bucket["keywords"].update(matched_keywords)
            bucket["scores"].extend(scores)

    return _grep_hits_from_matches(bundle, matches)


def grep_search(
    bundle: SearchBundle,
    keywords: list[str],
    kb: str | None = None,
    allowed_sources: set[str] | None = None,
) -> list[SearchHit]:
    keywords = _dedupe_strings(keywords)
    if not keywords:
        return []

    sources = _bundle_sources(bundle, kb=kb, allowed_sources=allowed_sources)
    hits = _grep_search_with_rg(bundle, keywords, sources)
    if hits:
        return hits
    return _grep_search_python(bundle, keywords, sources)


def _symbol_match_score(record: dict[str, Any], query_terms: list[str]) -> tuple[float, bool]:
    exact_symbol = False
    best = 0.0
    names = {
        record.get("name", "").lower(),
        record.get("qualified_name", "").lower(),
    }
    for term in query_terms:
        lowered = term.lower().rstrip("()")
        if not lowered:
            continue
        if lowered in names:
            exact_symbol = True
            best = max(best, 1.5)
            continue
        if any(lowered == name.split(".")[-1] for name in names if name):
            exact_symbol = True
            best = max(best, 1.4)
            continue
        if any(lowered in name for name in names if name):
            best = max(best, 0.9)
    return best, exact_symbol


def ast_search(
    bundle: SearchBundle,
    query_plan: QueryPlan,
    kb: str | None = None,
    allowed_sources: set[str] | None = None,
) -> list[SearchHit]:
    if not bundle.symbol_index:
        return []

    query_terms = _dedupe_strings([*query_plan.symbols, *query_plan.keywords])
    if not query_terms:
        return []

    valid_sources = set(_bundle_sources(bundle, kb=kb, allowed_sources=allowed_sources))
    hits: list[SearchHit] = []
    for record in bundle.symbol_index:
        source = record.get("source", "")
        if valid_sources and source not in valid_sources:
            continue

        score, exact_symbol = _symbol_match_score(record, query_terms)
        if score <= 0:
            continue

        cache_path = bundle.cache_path_for(source)
        if cache_path is None or not cache_path.exists():
            continue
        lines = _read_cached_text(str(cache_path)).splitlines()
        line_start = int(record.get("line_start") or 1)
        line_end = int(record.get("line_end") or line_start)
        capped_end = min(line_end, line_start + 119, len(lines))
        snippet = "\n".join(lines[line_start - 1 : capped_end]).strip()
        if not snippet:
            continue
        hits.append(
            SearchHit(
                source=source,
                match_kind="ast",
                snippet=snippet,
                score=score,
                line_start=line_start,
                line_end=capped_end,
                metadata={
                    "kind": record.get("kind", ""),
                    "signature": record.get("signature", ""),
                    "exact_symbol": exact_symbol,
                },
            )
        )

    return sorted(hits, key=lambda item: (item.score, -int(item.line_start or 0)), reverse=True)


def _vector_hit_from_document(doc: Document, raw_score: float | None = None) -> SearchHit:
    meta = doc.metadata
    return SearchHit(
        source=_normalize_source_path(meta.get("source", "")),
        match_kind="vector",
        snippet=doc.page_content,
        score=1.0 if raw_score is None else 1.0 / (1.0 + max(raw_score, 0.0)),
        line_start=meta.get("line_start"),
        line_end=meta.get("line_end"),
        metadata={"time_range": meta.get("time_range", "")},
    )


def _vector_search_raw(
    bundle: SearchBundle,
    query: str,
    kb: str | None = None,
    fetch_k: int = 24,
) -> list[SearchHit]:
    filter_dict = {"kb": kb} if kb is not None else None
    capped_fetch = min(max(1, fetch_k), bundle.vectorstore.index.ntotal or fetch_k)
    try:
        raw_results = bundle.vectorstore.similarity_search_with_score(
            query,
            k=capped_fetch,
            filter=filter_dict,
            fetch_k=capped_fetch,
        )
        return [_vector_hit_from_document(doc, score) for doc, score in raw_results]
    except Exception:
        docs = bundle.vectorstore.similarity_search(
            query,
            k=capped_fetch,
            filter=filter_dict,
            fetch_k=capped_fetch,
        )
        return [_vector_hit_from_document(doc) for doc in docs]


def vector_search(
    bundle: SearchBundle,
    query: str,
    kb: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    candidate_sources: set[str] | None = None,
) -> list[SearchHit]:
    if bundle.vectorstore.index.ntotal == 0:
        return []

    fetch_k = max(top_k * 4, len(candidate_sources or ()) * 3, 12)
    raw_hits = _vector_search_raw(bundle, query, kb=kb, fetch_k=fetch_k)

    filtered: list[SearchHit] = []
    seen: set[tuple[str, int | None, int | None, str]] = set()
    for hit in raw_hits:
        if candidate_sources is not None and hit.source not in candidate_sources:
            continue
        key = hit.dedupe_key()
        if key in seen:
            continue
        seen.add(key)
        filtered.append(hit)
        if len(filtered) >= top_k:
            break

    if candidate_sources and len(filtered) < min(2, top_k):
        for hit in raw_hits:
            key = hit.dedupe_key()
            if key in seen:
                continue
            seen.add(key)
            filtered.append(hit)
            if len(filtered) >= top_k:
                break

    return filtered


def _candidate_sources_from_hits(*hit_lists: list[SearchHit]) -> list[str]:
    return _dedupe_strings(
        hit.source
        for hits in hit_lists
        for hit in hits
    )


def _merge_grouped_hits(*grouped_sets: dict[str, list[SearchHit]]) -> dict[str, list[SearchHit]]:
    merged = {match_kind: [] for match_kind in RRF_WEIGHTS}
    for grouped in grouped_sets:
        for match_kind in merged:
            merged[match_kind].extend(grouped.get(match_kind, []))
    return merged


def _heuristic_expand_candidate_sources(
    bundle: SearchBundle,
    seed_sources: list[str],
    kb: str | None = None,
    allowed_sources: set[str] | None = None,
) -> GraphExpansionResult:
    expanded = set(seed_sources)
    grouped: dict[Path, list[str]] = {}
    for source in seed_sources:
        source_path = Path(source)
        grouped.setdefault(source_path.parent, []).append(source_path.stem.lower())

    edge_reasons: list[dict[str, Any]] = []
    for source in _bundle_sources(bundle, kb=kb, allowed_sources=allowed_sources):
        if source in expanded:
            continue
        candidate = Path(source)
        sibling_stems = grouped.get(candidate.parent)
        if not sibling_stems:
            continue
        stem = candidate.stem.lower()
        stem_tokens = _stem_tokens(stem)
        for sibling in sibling_stems:
            sibling_tokens = _stem_tokens(sibling)
            shared_tokens = stem_tokens & sibling_tokens
            if sibling in stem or stem in sibling or len(shared_tokens) >= 2:
                expanded.add(source)
                edge_reasons.append(
                    {
                        "from": str(candidate.parent / sibling),
                        "to": source,
                        "kind": "same_series",
                        "reason": ",".join(sorted(shared_tokens)) or sibling,
                        "hop": 1,
                    }
                )
                break

    return GraphExpansionResult(
        sources=expanded,
        seed_sources=seed_sources,
        expanded_sources=[source for source in expanded if source not in seed_sources],
        edge_reasons=edge_reasons,
        hops=1 if edge_reasons else 0,
        strategy="heuristic",
    )


def _document_graph_expand_candidate_sources(
    bundle: SearchBundle,
    ordered_seeds: list[str],
    valid_sources: set[str] | None,
    max_hops: int,
    max_extra_sources: int,
) -> GraphExpansionResult:
    seen = set(ordered_seeds)
    expanded_sources: list[str] = []
    edge_reasons: list[dict[str, Any]] = []
    queue: deque[tuple[str, int]] = deque((source, 0) for source in ordered_seeds)
    max_seen = len(ordered_seeds) + max_extra_sources

    while queue and len(seen) < max_seen:
        current, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for edge in bundle.graph_neighbors.get(current, []):
            if not isinstance(edge, dict):
                continue
            target = str(edge.get("target", "")).strip()
            if not target or target in seen:
                continue
            if valid_sources is not None and target not in valid_sources:
                continue
            seen.add(target)
            expanded_sources.append(target)
            hop = depth + 1
            edge_reasons.append(
                {
                    "from": current,
                    "to": target,
                    "kind": edge.get("kind", ""),
                    "reason": edge.get("reason", ""),
                    "hop": hop,
                }
            )
            if len(seen) >= max_seen:
                break
            queue.append((target, hop))

    return GraphExpansionResult(
        sources=seen,
        seed_sources=ordered_seeds,
        expanded_sources=expanded_sources,
        edge_reasons=edge_reasons,
        hops=max((item["hop"] for item in edge_reasons), default=0),
        strategy="document_graph",
    )


def _entity_graph_expand_candidate_sources(
    bundle: SearchBundle,
    ordered_seeds: list[str],
    valid_sources: set[str] | None,
    max_hops: int,
    max_extra_sources: int,
) -> GraphExpansionResult:
    seen = set(ordered_seeds)
    expanded_sources: list[str] = []
    edge_reasons: list[dict[str, Any]] = []
    bridge_entities: list[dict[str, Any]] = []
    seen_bridges: set[tuple[str, str]] = set()
    queue: deque[tuple[str, int]] = deque((source, 0) for source in ordered_seeds)
    max_seen = len(ordered_seeds) + max_extra_sources

    while queue and len(seen) < max_seen:
        current, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbor in _entity_file_neighbors(bundle, current, valid_sources=valid_sources):
            target = str(neighbor.get("to", "")).strip()
            if not target or target in seen:
                continue
            seen.add(target)
            expanded_sources.append(target)
            hop = depth + 1
            bridges = list(neighbor.get("bridges") or [])
            for bridge in bridges:
                _add_bridge_entity(bridge_entities, seen_bridges, bridge)
            edge_reasons.append(
                {
                    "from": current,
                    "to": target,
                    "kind": neighbor.get("kind", ""),
                    "reason": neighbor.get("reason", ""),
                    "hop": hop,
                    "bridges": bridges,
                }
            )
            if len(seen) >= max_seen:
                break
            queue.append((target, hop))

    return GraphExpansionResult(
        sources=seen,
        seed_sources=ordered_seeds,
        expanded_sources=expanded_sources,
        edge_reasons=edge_reasons,
        hops=max((item["hop"] for item in edge_reasons), default=0),
        strategy="entity_graph",
        bridge_entities=bridge_entities,
    )


def _expand_candidate_sources_detailed(
    bundle: SearchBundle,
    seed_sources: Iterable[str],
    kb: str | None = None,
    allowed_sources: set[str] | None = None,
    max_hops: int = 1,
    max_extra_sources: int = 12,
) -> GraphExpansionResult:
    valid_sources = (
        set(_bundle_sources(bundle, kb=kb, allowed_sources=allowed_sources))
        if kb is not None or allowed_sources is not None
        else None
    )
    ordered_seeds = [
        source
        for source in _dedupe_strings(seed_sources)
        if valid_sources is None or source in valid_sources
    ]
    if not ordered_seeds:
        return GraphExpansionResult(set(), [], [], [], 0)

    if bundle.entity_edges_by_source and bundle.entity_nodes_by_id:
        entity_result = _entity_graph_expand_candidate_sources(
            bundle,
            ordered_seeds,
            valid_sources=valid_sources,
            max_hops=max_hops,
            max_extra_sources=max_extra_sources,
        )
        if entity_result.expanded_sources:
            return entity_result

    if bundle.graph_neighbors:
        document_result = _document_graph_expand_candidate_sources(
            bundle,
            ordered_seeds,
            valid_sources=valid_sources,
            max_hops=max_hops,
            max_extra_sources=max_extra_sources,
        )
        if document_result.expanded_sources:
            return document_result

    return _heuristic_expand_candidate_sources(
        bundle,
        ordered_seeds,
        kb=kb,
        allowed_sources=allowed_sources,
    )


def _wiki_query_terms(question: str, query_plan: QueryPlan) -> tuple[list[str], list[str]]:
    phrases = _dedupe_strings(
        item
        for item in [question, query_plan.semantic_query]
        if isinstance(item, str) and len(item.strip()) >= 2
    )
    tokens = _dedupe_strings(
        cleaned
        for item in [*query_plan.symbols, *query_plan.keywords, *query_plan.path_globs]
        for cleaned in [str(item).strip().strip("*").strip(".,:;!?()[]{}")]
        if len(cleaned) >= 2 and cleaned.lower() not in GRAPH_STOPWORDS
    )
    return phrases, tokens


def _wiki_kind_priority(kind: str) -> int:
    return {"query": 4, "entity": 3, "community": 2, "file": 1, "index": 0}.get(kind, 0)


def _wiki_source_ref_matches_query(source: str, query_plan: QueryPlan) -> bool:
    source_lower = source.lower()
    basename_lower = Path(source).name.lower()
    specific_terms = _dedupe_strings(
        term.lower()
        for term in [*query_plan.keywords, *query_plan.symbols]
        if isinstance(term, str) and _looks_like_explicit_source_term(term)
    )
    specific_patterns = _dedupe_strings(
        pattern.lower()
        for pattern in query_plan.path_globs
        if isinstance(pattern, str)
        and pattern.strip()
        and not _is_extension_only_glob(pattern)
        and _looks_like_explicit_source_term(pattern.strip("*"))
    )
    if not specific_terms and not specific_patterns:
        return True

    for term in specific_terms:
        if term in source_lower or term in basename_lower:
            return True

    for pattern in specific_patterns:
        stripped = pattern.strip("*")
        if fnmatch.fnmatch(source_lower, pattern) or fnmatch.fnmatch(basename_lower, pattern):
            return True
        if stripped and (stripped in source_lower or stripped in basename_lower):
            return True

    return False


def _wiki_search(
    bundle: SearchBundle,
    question: str,
    query_plan: QueryPlan,
    kb: str | None = None,
    allowed_sources: set[str] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    if not bundle.wiki_pages:
        return []

    valid_sources = (
        set(_bundle_sources(bundle, kb=kb, allowed_sources=allowed_sources))
        if kb is not None or allowed_sources is not None
        else None
    )
    phrases, tokens = _wiki_query_terms(question, query_plan)
    if not phrases and not tokens:
        return []

    hits: list[dict[str, Any]] = []
    for page in bundle.wiki_pages:
        page_kind = str(page.get("kind", "") or "")
        page_title = str(page.get("title", "") or "")
        page_text = str(page.get("text", "") or "")
        source_refs = [
            source
            for source in page.get("source_refs", [])
            if isinstance(source, str) and source
        ]
        if valid_sources is not None:
            source_refs = [source for source in source_refs if source in valid_sources]
        if source_refs:
            source_refs = [source for source in source_refs if _wiki_source_ref_matches_query(source, query_plan)]
        if page_kind != "index" and not source_refs:
            continue

        lowered_text = page_text.lower()
        lowered_title = page_title.lower()
        matched_terms: list[str] = []
        score = 0.0

        for phrase in phrases:
            lowered_phrase = phrase.lower()
            if lowered_phrase not in lowered_text:
                continue
            matched_terms.append(phrase)
            score += 3.0 if page_kind == "query" else 2.0

        for token in tokens:
            lowered_token = token.lower()
            if lowered_token not in lowered_text:
                continue
            matched_terms.append(token)
            score += 1.5 if lowered_token in lowered_title else 1.0

        if page_kind == "file" and source_refs:
            page_source = source_refs[0]
            lowered_source = page_source.lower()
            lowered_basename = Path(page_source).name.lower()
            for pattern in query_plan.path_globs:
                normalized_pattern = pattern.lower()
                if fnmatch.fnmatch(lowered_source, normalized_pattern) or fnmatch.fnmatch(
                    lowered_basename, normalized_pattern
                ):
                    matched_terms.append(pattern)
                    score += 1.5
                    break

        deduped_terms = _dedupe_strings(matched_terms)
        if score <= 0 or not deduped_terms:
            continue

        hits.append(
            {
                "kind": page_kind,
                "title": page_title,
                "relpath": str(page.get("relpath", "") or ""),
                "score": round(score, 3),
                "matched_terms": deduped_terms,
                "source_refs": source_refs,
            }
        )

    hits.sort(
        key=lambda item: (
            -float(item.get("score", 0.0) or 0.0),
            -_wiki_kind_priority(str(item.get("kind", "") or "")),
            len(item.get("source_refs", [])),
            str(item.get("relpath", "") or ""),
        )
    )
    return hits[:limit]


def _run_search_step(
    question: str,
    bundle: SearchBundle,
    query_plan: QueryPlan,
    kb: str | None,
    top_k: int,
    allowed_sources: set[str] | None = None,
    step_name: str = "step1",
    graph_max_hops: int = 1,
    graph_max_extra_sources: int = 12,
) -> SearchStepResult:
    wiki_hits = _wiki_search(
        bundle,
        question=question,
        query_plan=query_plan,
        kb=kb,
        allowed_sources=allowed_sources,
    )
    wiki_seed_sources = _dedupe_strings(
        source
        for hit in wiki_hits
        for source in hit.get("source_refs", [])
        if isinstance(source, str) and source
    )
    wiki_scope = set(wiki_seed_sources)
    glob_hits = glob_search(bundle, query_plan.path_globs, kb=kb, allowed_sources=allowed_sources)
    glob_scope = wiki_scope | set(_candidate_sources_from_hits(glob_hits))
    grep_scope = glob_scope or allowed_sources
    grep_hits = grep_search(
        bundle,
        keywords=[*query_plan.keywords, *query_plan.symbols],
        kb=kb,
        allowed_sources=grep_scope,
    )
    grep_fallback_used = False
    should_retry_grep = not grep_hits and bool(glob_scope)
    if should_retry_grep:
        grep_hits = grep_search(
            bundle,
            keywords=[*query_plan.keywords, *query_plan.symbols],
            kb=kb,
            allowed_sources=allowed_sources,
        )
        grep_fallback_used = True
    ast_hits = ast_search(bundle, query_plan, kb=kb, allowed_sources=allowed_sources)
    graph_seed_sources = _dedupe_strings(
        [
            *wiki_seed_sources,
            *_candidate_sources_from_hits(glob_hits[:3], grep_hits[:3], ast_hits[:3]),
        ]
    )
    graph_expansion = _expand_candidate_sources_detailed(
        bundle,
        graph_seed_sources,
        kb=kb,
        allowed_sources=allowed_sources,
        max_hops=graph_max_hops,
        max_extra_sources=graph_max_extra_sources,
    )
    narrowed_vector_scope = wiki_scope | set(_candidate_sources_from_hits(glob_hits, grep_hits[:4], ast_hits[:4]))
    if graph_expansion.sources:
        narrowed_vector_scope = narrowed_vector_scope | graph_expansion.sources
    if not narrowed_vector_scope:
        narrowed_vector_scope = allowed_sources or set()

    vector_hits = vector_search(
        bundle,
        query=query_plan.semantic_query or question,
        kb=kb,
        top_k=max(top_k * 2, 8),
        candidate_sources=narrowed_vector_scope or None,
    )

    grouped = {
        "glob": glob_hits,
        "grep": grep_hits,
        "ast": ast_hits,
        "vector": vector_hits,
    }
    fused = _finalize_hits(grouped, top_k=top_k)
    trace = {
        "step": step_name,
        "query_plan": {
            "symbols": query_plan.symbols,
            "keywords": query_plan.keywords,
            "path_globs": query_plan.path_globs,
            "semantic_query": query_plan.semantic_query,
            "reason": query_plan.reason,
        },
        "retrievers": {name: len(hits) for name, hits in grouped.items()},
        "candidate_scope": sorted(narrowed_vector_scope) if narrowed_vector_scope else [],
        "wiki_hits": wiki_hits,
        "wiki_scope": wiki_seed_sources,
        "grep_fallback_used": grep_fallback_used,
        "graph_strategy": graph_expansion.strategy,
        "graph_seed_sources": graph_expansion.seed_sources,
        "graph_expanded_sources": graph_expansion.expanded_sources,
        "graph_edge_reasons": graph_expansion.edge_reasons,
        "graph_bridge_entities": graph_expansion.bridge_entities,
        "graph_hops": graph_expansion.hops,
        "top_sources": [hit.source for hit in fused[:3]],
    }
    return SearchStepResult(grouped_hits=grouped, hits=fused, trace=trace)


def _should_stop_after_first_step(hits: list[SearchHit]) -> tuple[bool, str]:
    if not hits:
        return False, "no_hits"

    top_three_sources = {hit.source for hit in hits[:3]}
    if len(top_three_sources) >= 2:
        return True, "top3_cover_two_files"

    top_hit = hits[0]
    if top_hit.metadata.get("exact_symbol"):
        return True, "exact_symbol_hit"

    if top_hit.match_kind in {"ast", "grep"}:
        second_score = hits[1].score if len(hits) > 1 else 0.0
        if top_hit.score >= second_score + 0.1:
            return True, "dominant_exact_hit"

    return False, "needs_followup"


def _extract_json_blob(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    match = re.search(r"\{.*\}", stripped, re.S)
    if not match:
        raise ValueError("planner output missing JSON object")
    data = json.loads(match.group(0))
    return data if isinstance(data, dict) else {}


def _planner_prompt(question: str, hits: list[SearchHit]) -> str:
    evidence_lines = []
    for hit in hits[:3]:
        location = _line_range_label(hit.line_start, hit.line_end)
        label = f"{hit.source}:{location}" if location else hit.source
        snippet = hit.snippet.splitlines()
        preview = "\n".join(snippet[:6]).strip()
        evidence_lines.append(f"[{label}]\n{preview}")
    evidence = "\n\n".join(evidence_lines) or "无"
    return f"""你是检索规划器。请根据用户问题和首轮命中结果，给出第二轮更精确的检索计划。

只返回一个 JSON 对象，不要输出任何额外说明，字段必须是：
- symbols: string[]
- keywords: string[]
- path_globs: string[]
- semantic_query: string
- reason: string

要求：
1. symbols 只放函数、类、配置键、import 名字等精确标识符
2. keywords 放适合 grep 的短语
3. path_globs 只放文件名/目录/扩展名模式
4. semantic_query 仍然是适合向量召回的完整问题
5. 不允许输出空对象；如果不确定，沿用问题里的原词

用户问题：
{question}

首轮命中：
{evidence}
"""


def _call_retrieval_planner(
    question: str,
    hits: list[SearchHit],
    llm_api_key: str,
    llm_model: str,
    llm_base_url: str,
) -> QueryPlan | None:
    if not llm_api_key.strip():
        return None
    try:
        llm = make_llm(
            api_key=llm_api_key,
            model=llm_model,
            base_url=llm_base_url,
            temperature=0.0,
        )
        response = llm.invoke(_planner_prompt(question, hits))
        content = _chunk_to_text(response)
        payload = _extract_json_blob(content)
        symbols = payload.get("symbols") or []
        keywords = payload.get("keywords") or []
        path_globs = payload.get("path_globs") or []
        semantic_query = payload.get("semantic_query") or question
        if not any([symbols, keywords, path_globs, semantic_query]):
            return None
        return QueryPlan(
            symbols=_dedupe_strings(str(item) for item in symbols),
            keywords=_dedupe_strings(str(item) for item in keywords),
            path_globs=_dedupe_strings(str(item) for item in path_globs),
            semantic_query=str(semantic_query).strip() or question,
            reason=str(payload.get("reason") or "llm planner"),
        )
    except Exception:
        return None


def _expand_candidate_sources(
    bundle: SearchBundle,
    seed_sources: Iterable[str],
    kb: str | None = None,
    allowed_sources: set[str] | None = None,
    max_hops: int = 1,
    max_extra_sources: int = 12,
) -> set[str]:
    return _expand_candidate_sources_detailed(
        bundle,
        seed_sources,
        kb=kb,
        allowed_sources=allowed_sources,
        max_hops=max_hops,
        max_extra_sources=max_extra_sources,
    ).sources


def _merge_plans(base: QueryPlan, followup: QueryPlan | None, question: str) -> QueryPlan:
    if followup is None:
        return base
    return QueryPlan(
        symbols=_dedupe_strings([*base.symbols, *followup.symbols]),
        keywords=_dedupe_strings([*base.keywords, *followup.keywords]),
        path_globs=_dedupe_strings([*base.path_globs, *followup.path_globs]),
        semantic_query=followup.semantic_query.strip() or question,
        reason=followup.reason or base.reason,
    )


def _sources_from_hits(results: list[SearchHit]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    for hit in results:
        key = (hit.source, hit.line_start, hit.line_end)
        if key in seen:
            continue
        seen.add(key)
        time_range = hit.metadata.get("time_range") or _line_range_label(hit.line_start, hit.line_end)
        snippet = hit.snippet
        sources.append(
            {
                "source": hit.source,
                "time_range": time_range,
                "snippet": snippet[:240] + "..." if len(snippet) > 240 else snippet,
                "match_kind": hit.match_kind,
                "line_start": hit.line_start,
                "line_end": hit.line_end,
            }
        )
    return sources


def _collect_bridge_entities(search_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bridge_entities: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for step in search_trace:
        raw_entities = step.get("graph_bridge_entities", [])
        if not isinstance(raw_entities, list):
            continue
        for raw_entity in raw_entities:
            if not isinstance(raw_entity, dict):
                continue
            _add_bridge_entity(bridge_entities, seen, raw_entity)
    return bridge_entities


def _collect_wiki_trace(search_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wiki_trace: list[dict[str, Any]] = []
    for step in search_trace:
        wiki_hits = step.get("wiki_hits", [])
        wiki_scope = step.get("wiki_scope", [])
        if not isinstance(wiki_hits, list):
            wiki_hits = []
        if not isinstance(wiki_scope, list):
            wiki_scope = []
        wiki_trace.append(
            {
                "step": str(step.get("step", "") or ""),
                "hit_count": len(wiki_hits),
                "scope_count": len(wiki_scope),
                "hits": wiki_hits,
                "scope": wiki_scope,
            }
        )
    return wiki_trace


def _bundle_artifacts_summary(search_bundle: SearchBundle) -> dict[str, Any]:
    wiki_pages = [
        {
            "kind": str(page.get("kind", "") or ""),
            "title": str(page.get("title", "") or ""),
            "relpath": str(page.get("relpath", "") or ""),
            "source_refs": [
                str(source)
                for source in page.get("source_refs", [])
                if isinstance(source, str) and source
            ],
        }
        for page in search_bundle.wiki_pages
        if isinstance(page, dict)
    ]
    return {
        "wiki_pages": wiki_pages,
        "community_pages": [page for page in wiki_pages if page.get("kind") == "community"],
        "entity_pages": [page for page in wiki_pages if page.get("kind") == "entity"],
        "graph_report_path": str(
            search_bundle.manifest.get("graph_report_file", f"{REPORTS_DIRNAME}/{GRAPH_REPORT_FILENAME}")
            or f"{REPORTS_DIRNAME}/{GRAPH_REPORT_FILENAME}"
        ),
        "community_index_path": str(
            search_bundle.manifest.get("community_index_file", COMMUNITY_INDEX_FILENAME)
            or COMMUNITY_INDEX_FILENAME
        ),
        "lint_report_path": str(
            search_bundle.manifest.get("lint_report_file", LINT_REPORT_FILENAME) or LINT_REPORT_FILENAME
        ),
    }


def _build_context_and_sources(results: list[SearchHit]) -> tuple[str, list[dict[str, Any]]]:
    context_parts = []
    for index, hit in enumerate(results, start=1):
        location = _line_range_label(hit.line_start, hit.line_end)
        title = hit.source if not location else f"{hit.source}:{location}"
        context_parts.append(f"[证据{index}] {title}\n{hit.snippet}")
    context = "\n\n".join(context_parts) if context_parts else "未找到直接匹配的参考资料。"
    return context, _sources_from_hits(results)


def retrieve(
    question: str,
    search_bundle: SearchBundle,
    kb: str | None = None,
    mode: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    llm_api_key: str = "",
    llm_model: str = "",
    llm_base_url: str = "",
) -> dict[str, Any]:
    search_mode = _search_mode(mode)
    base_plan = _extract_query_plan(question)
    trace: list[dict[str, Any]] = []

    if search_mode == "vector":
        vector_hits = vector_search(search_bundle, question, kb=kb, top_k=top_k)
        context, sources = _build_context_and_sources(vector_hits)
        trace.append(
            {
                "step": "step1",
                "query_plan": {
                    "symbols": [],
                    "keywords": [],
                    "path_globs": [],
                    "semantic_query": question,
                    "reason": "vector only",
                },
                "retrievers": {"vector": len(vector_hits)},
                "candidate_scope": [],
                "top_sources": [hit.source for hit in vector_hits[:3]],
                "graph_strategy": "disabled",
                "graph_bridge_entities": [],
                "stopped": True,
                "stop_reason": "vector_mode",
            }
        )
        return {
            "hits": vector_hits,
            "context": context,
            "sources": sources,
            "search_trace": trace,
            "wiki_trace": _collect_wiki_trace(trace),
            "bridge_entities": [],
        }

    step1_result = _run_search_step(
        question=question,
        bundle=search_bundle,
        query_plan=base_plan,
        kb=kb,
        top_k=top_k,
        allowed_sources=None,
        step_name="step1",
    )
    stopped, reason = _should_stop_after_first_step(step1_result.hits)
    step1_result.trace["stopped"] = stopped or search_mode != "agentic" or _search_max_steps() <= 1
    step1_result.trace["stop_reason"] = reason if step1_result.trace["stopped"] else "followup_requested"
    trace.append(step1_result.trace)

    final_hits = step1_result.hits
    if search_mode == "agentic" and not stopped and _search_max_steps() > 1:
        prefilter_expansion = _expand_candidate_sources_detailed(
            search_bundle,
            _dedupe_strings(hit.source for hit in step1_result.hits),
            kb=kb,
            max_hops=2,
            max_extra_sources=20,
        )
        planner_plan = _call_retrieval_planner(
            question=question,
            hits=step1_result.hits,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
        )
        merged_plan = _merge_plans(base_plan, planner_plan, question)
        step2_result = _run_search_step(
            question=question,
            bundle=search_bundle,
            query_plan=merged_plan,
            kb=kb,
            top_k=top_k,
            allowed_sources=prefilter_expansion.sources or None,
            step_name="step2",
            graph_max_hops=2,
            graph_max_extra_sources=20,
        )
        step2_result.trace["planner_used"] = planner_plan is not None
        step2_result.trace["prefilter_graph_seed_sources"] = prefilter_expansion.seed_sources
        step2_result.trace["prefilter_graph_expanded_sources"] = prefilter_expansion.expanded_sources
        step2_result.trace["prefilter_graph_edge_reasons"] = prefilter_expansion.edge_reasons
        step2_result.trace["prefilter_graph_strategy"] = prefilter_expansion.strategy
        step2_result.trace["prefilter_graph_bridge_entities"] = prefilter_expansion.bridge_entities
        step2_result.trace["prefilter_graph_hops"] = prefilter_expansion.hops
        step2_result.trace["stopped"] = True
        step2_result.trace["stop_reason"] = "bounded_agentic_complete"
        trace.append(step2_result.trace)
        final_hits = _finalize_hits(
            _merge_grouped_hits(step1_result.grouped_hits, step2_result.grouped_hits),
            top_k=top_k,
        )

    context, sources = _build_context_and_sources(final_hits)
    bridge_entities = _collect_bridge_entities(trace)
    return {
        "hits": final_hits,
        "context": context,
        "sources": sources,
        "search_trace": trace,
        "wiki_trace": _collect_wiki_trace(trace),
        "bridge_entities": bridge_entities,
    }


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


def ask_stream(
    question: str,
    search_bundle: SearchBundle,
    llm_api_key: str,
    llm_model: str,
    llm_base_url: str,
    top_k: int = DEFAULT_TOP_K,
    kb: str | None = None,
    search_mode: str | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    retrieval = retrieve(
        question=question,
        search_bundle=search_bundle,
        kb=kb,
        mode=search_mode,
        top_k=top_k,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
    )

    llm = make_llm(api_key=llm_api_key, model=llm_model, base_url=llm_base_url)
    prompt = _PROMPT_TEMPLATE.format(context=retrieval["context"], question=question)

    def answer_stream():
        for chunk in llm.stream(prompt):
            text = _chunk_to_text(chunk)
            if text:
                yield text

    result: dict[str, Any] = {
        "answer_stream": answer_stream(),
        "sources": retrieval["sources"],
    }
    if debug:
        wiki_trace = retrieval.get("wiki_trace")
        if not isinstance(wiki_trace, list):
            wiki_trace = _collect_wiki_trace(retrieval.get("search_trace", []))
        result["search_trace"] = retrieval["search_trace"]
        result["wiki_trace"] = wiki_trace
        result["bridge_entities"] = retrieval.get("bridge_entities", [])
        result["artifacts"] = _bundle_artifacts_summary(search_bundle)
    return result
