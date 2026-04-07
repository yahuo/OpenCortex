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
import configparser
import fnmatch
import json
import os
import posixpath
import re
import shutil
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

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
DEFAULT_SEARCH_MODE = "hybrid"
SEARCH_MODES = {"vector", "hybrid", "agentic"}
SEARCH_MAX_STEPS_DEFAULT = 2
NORMALIZED_TEXT_DIRNAME = "normalized_texts"
SYMBOL_INDEX_FILENAME = "symbol_index.jsonl"
DOCUMENT_GRAPH_FILENAME = "document_graph.json"
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
    normalized_text: str
    chunks: list[ChunkSpec]
    symbols: list[dict[str, Any]]


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
    chunk_size: int = GENERIC_CHUNK_SIZE,
    overlap_lines: int = GENERIC_CHUNK_OVERLAP_LINES,
    label: str = "",
) -> list[ChunkSpec]:
    if not lines:
        return []

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


def _ensure_chunk_size(chunks: list[ChunkSpec]) -> list[ChunkSpec]:
    results: list[ChunkSpec] = []
    for chunk in chunks:
        if len(chunk.text) <= GENERIC_CHUNK_SIZE:
            results.append(chunk)
            continue
        lines = chunk.text.splitlines()
        start_line = chunk.line_start or 1
        results.extend(
            _split_lines_into_chunks(
                lines=lines,
                start_line=start_line,
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


def _parse_md_messages_text(content: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    current_sender: str | None = None
    current_ts: datetime | None = None
    current_lines: list[str] = []
    current_start_line: int | None = None
    current_end_line: int | None = None

    def flush():
        if current_sender is None or current_start_line is None:
            return
        body = "\n".join(current_lines).strip()
        if not body:
            return
        messages.append(
            {
                "sender": current_sender,
                "timestamp": current_ts,
                "content": body,
                "line_start": current_start_line,
                "line_end": current_end_line or current_start_line,
            }
        )

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
        if stripped and stripped not in ("---",) and not stripped.startswith("> 导出时间"):
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
    parts = Path(rel_path).parts
    if len(parts) > 1:
        return parts[0]
    return ""


def _process_source_file(root: Path, file_path: Path) -> IndexedFile | None:
    rel_path = str(file_path.relative_to(root))
    kb = _extract_kb(rel_path)
    suffix = file_path.suffix.lower()

    try:
        if suffix in _BINARY_SUFFIXES:
            raw_text = _convert_binary_to_markdown(file_path)
        else:
            raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    normalized_raw = _normalize_text(raw_text)
    if not normalized_raw:
        return None

    symbols: list[dict[str, Any]] = []
    if suffix in MARKDOWN_SUFFIXES or suffix in _BINARY_SUFFIXES:
        messages = _parse_md_messages_text(normalized_raw) if suffix in MARKDOWN_SUFFIXES else []
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
            chunks = _chunk_markdown_by_heading(normalized_raw)
        normalized_text = normalized_raw
    elif suffix in STRUCTURED_SUFFIXES:
        normalized_text = _normalize_structured_text(normalized_raw, suffix)
        chunks = _chunk_structured_text(normalized_text)
    elif suffix in PYTHON_SUFFIXES:
        normalized_text = normalized_raw
        chunks = _chunk_python_code(normalized_raw)
        symbols = _extract_python_symbols(normalized_raw, rel_path)
    else:
        normalized_text = normalized_raw
        chunks = _split_lines_into_chunks(normalized_text.splitlines())

    if not chunks:
        chunks = _split_lines_into_chunks(normalized_text.splitlines())
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

    if indexed.suffix in STRUCTURED_SUFFIXES:
        for line in indexed.normalized_text.splitlines():
            if " = " not in line:
                continue
            key = line.split(" = ", 1)[0].strip()
            if re.fullmatch(r"[A-Za-z0-9_.-]{3,}", key):
                tokens.add(key.lower())
            for piece in re.split(r"[.\[\]-]+", key):
                token = _normalize_graph_token(piece)
                if token:
                    tokens.add(token)

    for match in CODE_SPAN_RE.findall(indexed.normalized_text):
        token = _normalize_graph_token(match)
        if token:
            tokens.add(token)

    for match in CALL_RE.findall(indexed.normalized_text):
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

    clean = clean.split("#", 1)[0].split("?", 1)[0].replace("\\", "/")
    if not clean:
        return None
    if clean in source_lookup:
        return clean

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
    sources = sorted(indexed.rel_path for indexed in indexed_files)
    source_lookup = set(sources)
    basename_lookup: dict[str, list[str]] = {}
    for source in sources:
        basename_lookup.setdefault(Path(source).name, []).append(source)

    edges_by_source: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {
        source: {} for source in sources
    }

    for indexed in indexed_files:
        for kind, reference in _iter_local_path_references(indexed.normalized_text):
            target = _resolve_document_reference(
                indexed.rel_path,
                reference,
                source_lookup,
                basename_lookup,
            )
            if target is None:
                continue
            _add_graph_edge(edges_by_source, indexed.rel_path, target, kind, reference)

    token_sources: dict[str, set[str]] = {}
    for indexed in indexed_files:
        for token in _extract_shared_tokens(indexed):
            token_sources.setdefault(token, set()).add(indexed.rel_path)

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


# ─────────────────────────────────────────────────────────
# 向量库 / SearchBundle 操作
# ─────────────────────────────────────────────────────────
def _write_index_artifacts(
    persist_path: Path,
    indexed_files: list[IndexedFile],
    embed_model: str,
    source_dir: str,
    total_chunks: int,
) -> dict[str, Any]:
    normalized_dir = persist_path / NORMALIZED_TEXT_DIRNAME
    if normalized_dir.exists():
        shutil.rmtree(normalized_dir)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    symbol_records: list[dict[str, Any]] = []
    manifest = {
        "build_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "embed_model": embed_model,
        "total_chunks": total_chunks,
        "kb_enabled": True,
        "source_dir": str(Path(source_dir).expanduser()),
        "normalized_text_dir": NORMALIZED_TEXT_DIRNAME,
        "symbol_index_file": SYMBOL_INDEX_FILENAME,
        "document_graph_file": DOCUMENT_GRAPH_FILENAME,
        "search_mode_default": os.getenv("SEARCH_MODE", DEFAULT_SEARCH_MODE).strip() or DEFAULT_SEARCH_MODE,
        "files": [],
    }

    for indexed in indexed_files:
        normalized_rel = f"{indexed.rel_path}.txt"
        cache_path = normalized_dir / normalized_rel
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(indexed.normalized_text, encoding="utf-8")

        stat = indexed.file_path.stat()
        manifest["files"].append(
            {
                "name": indexed.rel_path,
                "kb": indexed.kb,
                "suffix": indexed.suffix,
                "size_kb": round(stat.st_size / 1024, 1),
                "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "chunks": len(indexed.chunks),
                "normalized_text": normalized_rel,
            }
        )
        symbol_records.extend(indexed.symbols)

    symbol_index_path = persist_path / SYMBOL_INDEX_FILENAME
    if symbol_records:
        symbol_index_path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in symbol_records),
            encoding="utf-8",
        )
    elif symbol_index_path.exists():
        symbol_index_path.unlink()

    document_graph = _build_document_graph(indexed_files)
    document_graph_path = persist_path / DOCUMENT_GRAPH_FILENAME
    document_graph_path.write_text(
        json.dumps(document_graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (persist_path / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def build_vectorstore(
    md_dir: str,
    embed_api_key: str,
    embed_base_url: str = SILICONFLOW_BASE_URL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    persist_dir: str = DEFAULT_FAISS_DIR,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> SearchBundle:
    """
    构建（或重建）FAISS 向量索引并持久化到本地。
    """

    def _cb(current: int, total: int, message: str) -> None:
        if progress_callback:
            progress_callback(current, total, message)

    _cb(0, 1, "正在扫描目录并解析文本文件...")
    documents, indexed_files = _build_documents(md_dir)
    total = len(documents)

    if total == 0:
        raise ValueError(
            f"在 {md_dir} 中未找到可索引文本。支持后缀: {', '.join(sorted(SUPPORTED_TEXT_SUFFIXES))}"
        )

    _cb(0, total + 3, f"解析完毕，共 {len(indexed_files)} 个文件，{total} 个分片。")
    embeddings = make_embeddings(api_key=embed_api_key, base_url=embed_base_url, model=embed_model)

    batch_size = max(1, int(os.getenv("EMBED_BATCH_SIZE", "10")))
    batch_sleep_seconds = max(0.0, float(os.getenv("EMBED_BATCH_SLEEP_SECONDS", "0.5")))
    max_retries = max(0, int(os.getenv("EMBED_MAX_RETRIES", "8")))
    retry_base_seconds = max(0.5, float(os.getenv("EMBED_RETRY_BASE_SECONDS", "5")))

    vectorstore: FAISS | None = None
    total_steps = total + 3
    for i in range(0, total, batch_size):
        batch = documents[i : i + batch_size]
        for attempt in range(max_retries + 1):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.add_documents(batch)
                break
            except Exception as exc:
                if not _is_rate_limit_error(exc) or attempt >= max_retries:
                    raise
                delay = retry_base_seconds * (2**attempt)
                _cb(
                    i,
                    total_steps,
                    f"触发 embedding 限流，{delay:.1f}s 后重试第 {attempt + 1}/{max_retries} 次...",
                )
                time.sleep(delay)
        done = min(i + batch_size, total)
        _cb(done, total_steps, f"向量化中 {done}/{total}...")
        if batch_sleep_seconds > 0 and done < total:
            time.sleep(batch_sleep_seconds)

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(persist_path))  # type: ignore[union-attr]
    _cb(total + 1, total_steps, "正在写入检索辅助索引...")

    manifest = _write_index_artifacts(
        persist_path=persist_path,
        indexed_files=indexed_files,
        embed_model=embed_model,
        source_dir=md_dir,
        total_chunks=total,
    )
    _read_cached_text.cache_clear()
    _cb(total + 2, total_steps, "正在加载检索 bundle...")

    bundle = load_search_bundle(
        embed_api_key=embed_api_key,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        persist_dir=persist_dir,
    )
    if bundle is None:
        raise RuntimeError("索引文件写入成功，但 SearchBundle 回读失败。")

    _cb(total + 3, total_steps, f"✅ 索引构建完成，已保存到 {persist_dir}")
    bundle.manifest = manifest
    return bundle


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
    files = manifest.get("files", [])
    files_by_source = {entry.get("name", ""): entry for entry in files if entry.get("name")}
    normalized_text_dir = Path(persist_dir) / manifest.get("normalized_text_dir", NORMALIZED_TEXT_DIRNAME)
    symbol_index_path = Path(persist_dir) / manifest.get("symbol_index_file", SYMBOL_INDEX_FILENAME)
    document_graph_path = Path(persist_dir) / manifest.get("document_graph_file", DOCUMENT_GRAPH_FILENAME)

    symbol_index: list[dict[str, Any]] = []
    if symbol_index_path.exists():
        for line in symbol_index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                symbol_index.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    document_graph: dict[str, Any] = {"version": 1, "edge_count": 0, "neighbors": {}}
    if document_graph_path.exists():
        try:
            document_graph = json.loads(document_graph_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            document_graph = {"version": 1, "edge_count": 0, "neighbors": {}}
    graph_neighbors = document_graph.get("neighbors", {})
    if not isinstance(graph_neighbors, dict):
        graph_neighbors = {}

    source_dir = manifest.get("source_dir")
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

    raw_tokens = quoted + TOKEN_RE.findall(question)
    path_globs: list[str] = []
    symbols: list[str] = []
    keywords: list[str] = []

    for token in raw_tokens:
        clean = token.strip().strip(".,:;!?()[]{}")
        if not clean:
            continue
        keywords.append(clean)

        if PATHISH_RE.fullmatch(clean) or "/" in clean or "*" in clean:
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

        if "." in stripped_call and all(part for part in stripped_call.split(".")):
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
        source=meta.get("source", ""),
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


def _candidate_sources_from_hits(*hit_lists: list[SearchHit]) -> set[str]:
    sources: set[str] = set()
    for hits in hit_lists:
        for hit in hits:
            sources.add(hit.source)
    return sources


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

    if not bundle.graph_neighbors:
        return _heuristic_expand_candidate_sources(
            bundle,
            ordered_seeds,
            kb=kb,
            allowed_sources=allowed_sources,
        )

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

    if not expanded_sources:
        heuristic = _heuristic_expand_candidate_sources(
            bundle,
            ordered_seeds,
            kb=kb,
            allowed_sources=allowed_sources,
        )
        if heuristic.expanded_sources:
            return heuristic

    return GraphExpansionResult(
        sources=seen,
        seed_sources=ordered_seeds,
        expanded_sources=expanded_sources,
        edge_reasons=edge_reasons,
        hops=max((item["hop"] for item in edge_reasons), default=0),
    )


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
    glob_hits = glob_search(bundle, query_plan.path_globs, kb=kb, allowed_sources=allowed_sources)
    glob_scope = _candidate_sources_from_hits(glob_hits)
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
    graph_expansion = _expand_candidate_sources_detailed(
        bundle,
        _candidate_sources_from_hits(glob_hits[:3], grep_hits[:3], ast_hits[:3]),
        kb=kb,
        allowed_sources=allowed_sources,
        max_hops=graph_max_hops,
        max_extra_sources=graph_max_extra_sources,
    )
    narrowed_vector_scope = _candidate_sources_from_hits(glob_hits, grep_hits[:4], ast_hits[:4])
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
        "grep_fallback_used": grep_fallback_used,
        "graph_seed_sources": graph_expansion.seed_sources,
        "graph_expanded_sources": graph_expansion.expanded_sources,
        "graph_edge_reasons": graph_expansion.edge_reasons,
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
                "stopped": True,
                "stop_reason": "vector_mode",
            }
        )
        return {"hits": vector_hits, "context": context, "sources": sources, "search_trace": trace}

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
        step2_result.trace["prefilter_graph_hops"] = prefilter_expansion.hops
        step2_result.trace["stopped"] = True
        step2_result.trace["stop_reason"] = "bounded_agentic_complete"
        trace.append(step2_result.trace)
        final_hits = _finalize_hits(
            _merge_grouped_hits(step1_result.grouped_hits, step2_result.grouped_hits),
            top_k=top_k,
        )

    context, sources = _build_context_and_sources(final_hits)
    return {
        "hits": final_hits,
        "context": context,
        "sources": sources,
        "search_trace": trace,
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
        result["search_trace"] = retrieval["search_trace"]
    return result
