from __future__ import annotations

import ast
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import configparser
import fnmatch
import json
import os
from queue import Empty, Queue
import shutil
import tempfile
from threading import local
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    yaml = None
    _YAML_AVAILABLE = False

try:
    from markitdown import MarkItDown

    _MARKITDOWN_AVAILABLE = True
except ImportError:
    MarkItDown = None
    _MARKITDOWN_AVAILABLE = False

if TYPE_CHECKING:
    from ragbot import IndexedFile, SearchBundle


def _core():
    import ragbot as core

    return core


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


def _should_ignore_relative_path(rel_path: Path, extra_patterns: list[str]) -> bool:
    core = _core()
    parts = set(rel_path.parts)
    if parts & core.DEFAULT_IGNORED_DIRS:
        return True
    posix_path = rel_path.as_posix()
    for pattern in extra_patterns:
        if fnmatch.fnmatch(posix_path, pattern) or fnmatch.fnmatch(rel_path.name, pattern):
            return True
    return False


def _iter_supported_files(source_dir: Path) -> list[Path]:
    core = _core()
    files: list[Path] = []
    extra_patterns = core._load_extra_exclude_globs()

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
            if path.suffix.lower() in core.SUPPORTED_TEXT_SUFFIXES:
                files.append(path)

    return sorted(files)


def _normalize_structured_text(text: str, suffix: str) -> str:
    core = _core()
    try:
        if suffix == ".json":
            parsed = json.loads(text)
            return "\n".join(core._flatten_mapping(parsed))
        if suffix in {".yaml", ".yml"} and _YAML_AVAILABLE:
            parsed = yaml.safe_load(text)
            return "\n".join(core._flatten_mapping(parsed))
        if suffix == ".toml":
            parsed = tomllib.loads(text)
            return "\n".join(core._flatten_mapping(parsed))
        if suffix == ".ini":
            parser = configparser.ConfigParser()
            parser.optionxform = str
            parser.read_string(text)
            data: dict[str, Any] = {}
            for section in parser.sections():
                data[section] = dict(parser.items(section))
            return "\n".join(core._flatten_mapping(data))
    except Exception:
        pass
    return core._normalize_text(text)


def _chunk_structured_text(normalized_text: str) -> list[Any]:
    core = _core()
    return core._split_lines_into_chunks(normalized_text.splitlines())


def _convert_binary_to_markdown(file_path: Path) -> str:
    if not _MARKITDOWN_AVAILABLE:
        raise ImportError(
            f"处理 {file_path.suffix} 文件需要 markitdown 库，"
            "请运行: pip install 'markitdown[pdf,docx,xlsx]'"
        )
    md = MarkItDown()
    result = md.convert(str(file_path))
    return result.text_content


def _chunk_markdown_by_heading(text: str) -> list[Any]:
    core = _core()
    lines = text.splitlines()
    if not lines:
        return []

    heading_indexes = [
        idx for idx, line in enumerate(lines) if core.re.match(r"^\s{0,3}#{1,6}\s+", line)
    ]
    if not heading_indexes:
        return core._split_lines_into_chunks(lines)

    chunks: list[Any] = []
    starts = heading_indexes + [len(lines)]
    for current, next_index in zip(starts, starts[1:]):
        section_lines = lines[current:next_index]
        section_text = "\n".join(section_lines).strip()
        if not section_text:
            continue
        heading_line = section_lines[0].strip()
        label = heading_line.lstrip("#").strip() or f"section {len(chunks) + 1}"
        chunks.append(
            core.ChunkSpec(
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
                core.ChunkSpec(
                    text=preface_text,
                    line_start=1,
                    line_end=preface_end,
                    label="preface",
                ),
            )

    return core._ensure_chunk_size(chunks)


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
) -> Any:
    core = _core()
    if not lines or start_line is None:
        return None
    text = "\n".join(lines).strip()
    if not text:
        return None
    return core.ChunkSpec(
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
) -> Iterable[Any]:
    core = _core()
    chunk_size = core._configured_chunk_size() if chunk_size is None else max(1, chunk_size)
    overlap_lines = (
        core._configured_chunk_overlap_lines()
        if overlap_lines is None
        else max(0, overlap_lines)
    )
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


def _iter_markdown_heading_chunks_from_file(path: Path) -> Iterable[Any]:
    core = _core()
    heading_re = core.re.compile(r"^\s{0,3}#{1,6}\s+")
    chunk_size = core._configured_chunk_size()
    overlap_lines = core._configured_chunk_overlap_lines()
    segment_lines: list[str] = []
    segment_start_line: int | None = None
    segment_label = "preface"
    current_len = 0
    saw_heading = False

    def push_line(line_no: int, line: str) -> Iterable[Any]:
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

    def flush_segment() -> Iterable[Any]:
        nonlocal segment_lines, segment_start_line, current_len
        chunk = _emit_line_chunk(segment_lines, segment_start_line, label=segment_label)
        if chunk is not None:
            yield chunk
        segment_lines = []
        segment_start_line = None
        current_len = 0

    for line_no, line in enumerate(core._iter_text_file_lines(path), start=1):
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
    window_minutes: int = 30,
) -> Iterable[Any]:
    core = _core()
    current_sender: str | None = None
    current_ts = None
    current_lines: list[str] = []
    current_start_line: int | None = None
    current_end_line: int | None = None
    current_chunk: list[dict[str, Any]] = []
    window_start = None
    chunk_index = 0

    def flush_message() -> dict[str, Any] | None:
        return core._build_wechat_message(
            current_sender,
            current_ts,
            current_lines,
            current_start_line,
            current_end_line,
        )

    def emit_chunk(messages: list[dict[str, Any]]) -> Any:
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
        return core.ChunkSpec(
            text=text,
            line_start=messages[0].get("line_start"),
            line_end=messages[-1].get("line_end"),
            label=label,
        )

    for lineno, line in enumerate(core._iter_text_file_lines(path), start=1):
        match = core._HEADER_RE.match(line)
        if match:
            message = flush_message()
            if message is not None:
                ts = message.get("timestamp")
                if ts is None:
                    current_chunk.append(message)
                else:
                    if window_start is None:
                        window_start = ts
                    if ts - window_start <= core.timedelta(minutes=window_minutes):
                        current_chunk.append(message)
                    else:
                        chunk = emit_chunk(current_chunk)
                        if chunk is not None:
                            yield from core._ensure_chunk_size([chunk])
                        current_chunk = [message]
                        window_start = ts
            current_sender = match.group(1).strip()
            try:
                current_ts = core.datetime.strptime(match.group(2).strip(), "%Y-%m-%d %H:%M:%S")
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
            if window_start is None or ts - window_start <= core.timedelta(minutes=window_minutes):
                current_chunk.append(message)
            else:
                chunk = emit_chunk(current_chunk)
                if chunk is not None:
                    yield from core._ensure_chunk_size([chunk])
                current_chunk = [message]
                window_start = ts

    chunk = emit_chunk(current_chunk)
    if chunk is not None:
        yield from core._ensure_chunk_size([chunk])


def _iter_chunks_from_cached_file(
    path: Path,
    suffix: str,
    chunk_strategy: str,
) -> Iterable[Any]:
    core = _core()
    if chunk_strategy == "wechat_markdown":
        return core._iter_limited_chunks(
            core._iter_wechat_markdown_chunks_from_file(path),
            chunk_strategy=chunk_strategy,
        )
    if chunk_strategy == "markdown_heading":
        return core._iter_limited_chunks(
            core._iter_markdown_heading_chunks_from_file(path),
            chunk_strategy=chunk_strategy,
        )
    if chunk_strategy in {"structured", "generic"}:
        return core._iter_limited_chunks(
            core._iter_chunk_specs_from_line_stream(
                enumerate(core._iter_text_file_lines(path), start=1)
            ),
            chunk_strategy=chunk_strategy,
        )
    if chunk_strategy == "python":
        return iter(
            core._limit_chunk_specs(
                core._chunk_python_code(path.read_text(encoding="utf-8")),
                chunk_strategy=chunk_strategy,
            )
        )
    return iter(core._chunks_for_indexed_text(path.read_text(encoding="utf-8"), suffix, chunk_strategy))


def _chunk_python_code(text: str) -> list[Any]:
    core = _core()
    normalized = core._normalize_text(text)
    try:
        tree = ast.parse(normalized)
    except SyntaxError:
        return core._split_lines_into_chunks(normalized.splitlines())

    lines = normalized.splitlines()
    body_nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    if not body_nodes:
        return core._split_lines_into_chunks(lines)

    chunks: list[Any] = []
    preface_end = body_nodes[0].lineno - 1
    if preface_end > 0:
        chunks.extend(
            core._split_lines_into_chunks(lines[:preface_end], start_line=1, label="module prelude")
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
            core.ChunkSpec(
                text="\n".join(node_lines).strip(),
                line_start=start,
                line_end=end,
                label=label,
            )
        )

    return core._ensure_chunk_size(chunks)


def _extract_python_symbols(text: str, rel_path: str) -> list[dict[str, Any]]:
    core = _core()
    normalized = core._normalize_text(text)
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


def _prepare_indexed_content(
    rel_path: str,
    suffix: str,
    normalized_raw: str,
) -> tuple[str, list[Any], list[dict[str, Any]], str, int, int]:
    core = _core()
    symbols: list[dict[str, Any]] = []
    chunk_strategy = "generic"
    if suffix in core.MARKDOWN_SUFFIXES or suffix in core._BINARY_SUFFIXES:
        messages = core._parse_md_messages_text(normalized_raw) if suffix in core.MARKDOWN_SUFFIXES else []
        if messages:
            chunk_strategy = "wechat_markdown"
            chunk_specs: list[Any] = []
            for idx, chunk in enumerate(core._chunk_by_time_window(messages), start=1):
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
                    core.ChunkSpec(
                        text=text,
                        line_start=line_start,
                        line_end=line_end,
                        label=label,
                    )
                )
            chunks = core._ensure_chunk_size(chunk_specs)
        else:
            if any(core.re.match(r"^\s{0,3}#{1,6}\s+", line) for line in normalized_raw.splitlines()):
                chunk_strategy = "markdown_heading"
                chunks = core._chunk_markdown_by_heading(normalized_raw)
            else:
                chunks = core._split_lines_into_chunks(normalized_raw.splitlines())
        normalized_text = normalized_raw
    elif suffix in core.STRUCTURED_SUFFIXES:
        chunk_strategy = "structured"
        normalized_text = core._normalize_structured_text(normalized_raw, suffix)
        chunks = core._chunk_structured_text(normalized_text)
    elif suffix in core.PYTHON_SUFFIXES:
        chunk_strategy = "python"
        normalized_text = normalized_raw
        chunks = core._chunk_python_code(normalized_raw)
        symbols = core._extract_python_symbols(normalized_raw, rel_path)
    else:
        normalized_text = normalized_raw
        chunks = core._split_lines_into_chunks(normalized_text.splitlines())

    if not chunks:
        chunks = core._split_lines_into_chunks(normalized_text.splitlines())
    original_chunk_total = len(chunks)
    chunks = core._limit_chunk_specs(chunks, chunk_strategy=chunk_strategy)
    return (
        normalized_text,
        chunks,
        symbols,
        chunk_strategy,
        max(1, len(normalized_text.splitlines())),
        original_chunk_total,
    )


def _chunks_for_indexed_text(
    normalized_text: str,
    suffix: str,
    chunk_strategy: str = "",
) -> list[Any]:
    core = _core()
    if not normalized_text:
        return []
    strategy = chunk_strategy.strip()
    if strategy == "wechat_markdown" or ((suffix in core.MARKDOWN_SUFFIXES or suffix in core._BINARY_SUFFIXES) and not strategy):
        messages = core._parse_md_messages_text(normalized_text) if suffix in core.MARKDOWN_SUFFIXES else []
        if messages:
            chunk_specs: list[Any] = []
            for idx, chunk in enumerate(core._chunk_by_time_window(messages), start=1):
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
                    core.ChunkSpec(
                        text=text,
                        line_start=line_start,
                        line_end=line_end,
                        label=label,
                    )
                )
            chunks = core._ensure_chunk_size(chunk_specs)
        else:
            chunks = core._chunk_markdown_by_heading(normalized_text)
    elif strategy == "markdown_heading":
        chunks = core._chunk_markdown_by_heading(normalized_text)
    elif strategy == "structured" or suffix in core.STRUCTURED_SUFFIXES:
        chunks = core._chunk_structured_text(normalized_text)
    elif strategy == "python" or suffix in core.PYTHON_SUFFIXES:
        chunks = core._chunk_python_code(normalized_text)
    else:
        chunks = core._split_lines_into_chunks(normalized_text.splitlines())
    if chunks:
        return core._limit_chunk_specs(chunks, chunk_strategy=strategy)
    return core._limit_chunk_specs(
        core._split_lines_into_chunks(normalized_text.splitlines()),
        chunk_strategy=strategy,
    )


def _read_source_text(file_path: Path, suffix: str) -> str | None:
    core = _core()
    try:
        if suffix in core._BINARY_SUFFIXES:
            return core._convert_binary_to_markdown(file_path)
        return file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def _normalized_cache_path(cache_root: Path, rel_path: str) -> Path:
    return cache_root / f"{rel_path}.txt"


def _process_source_file(root: Path, file_path: Path) -> IndexedFile | None:
    core = _core()
    rel_path = core._normalize_source_path(file_path.relative_to(root).as_posix())
    kb = core._extract_kb(rel_path)
    suffix = file_path.suffix.lower()

    raw_text = core._read_source_text(file_path, suffix)
    if raw_text is None:
        return None

    normalized_raw = core._normalize_text(raw_text)
    if not normalized_raw:
        return None

    normalized_text, chunks, symbols, chunk_strategy, line_total, original_chunk_total = core._prepare_indexed_content(
        rel_path,
        suffix,
        normalized_raw,
    )
    if not chunks:
        return None

    return core.IndexedFile(
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
    core = _core()
    rel_path = core._normalize_source_path(file_path.relative_to(root).as_posix())
    kb = core._extract_kb(rel_path)
    suffix = file_path.suffix.lower()

    raw_text = core._read_source_text(file_path, suffix)
    if raw_text is None:
        return None

    normalized_raw = core._normalize_text(raw_text)
    if not normalized_raw:
        return None

    normalized_text, chunks, symbols, chunk_strategy, line_total, original_chunk_total = core._prepare_indexed_content(
        rel_path,
        suffix,
        normalized_raw,
    )
    if not chunks:
        return None

    cache_path = core._normalized_cache_path(cache_root, rel_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(normalized_text, encoding="utf-8")

    return core.IndexedFile(
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
    core = _core()
    root = Path(source_dir)
    files = core._iter_supported_files(root)
    documents: list[Document] = []
    indexed_files: list[IndexedFile] = []

    for file_path in files:
        processed = core._process_source_file(root, file_path)
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
                        "time_range": core._chunk_location_label(chunk, f"chunk {idx + 1}"),
                        "line_start": chunk.line_start,
                        "line_end": chunk.line_end,
                    },
                )
            )

    return documents, indexed_files


_ORIGINAL_BUILD_DOCUMENTS = _build_documents


def _index_source_files(source_dir: str, cache_root: Path) -> tuple[list[IndexedFile], int]:
    core = _core()
    root = Path(source_dir)
    indexed_files: list[IndexedFile] = []
    total_chunks = 0

    for file_path in core._iter_supported_files(root):
        processed = core._process_source_file_to_cache(root, file_path, cache_root)
        if processed is None:
            continue
        indexed_files.append(processed)
        total_chunks += processed.chunk_count

    return indexed_files, total_chunks


def _iter_documents_for_indexed_files(indexed_files: Iterable[IndexedFile]) -> Iterable[Document]:
    core = _core()
    for indexed in indexed_files:
        for idx, chunk in enumerate(indexed.iter_chunks()):
            yield Document(
                page_content=chunk.text,
                metadata={
                    "source": indexed.rel_path,
                    "kb": indexed.kb,
                    "chunk_index": idx,
                    "time_range": core._chunk_location_label(chunk, f"chunk {idx + 1}"),
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                },
            )


def _build_document_graph(indexed_files: list[IndexedFile]) -> dict[str, Any]:
    core = _core()
    sources = sorted(
        {
            normalized
            for indexed in indexed_files
            if (normalized := core._normalize_source_path(indexed.rel_path))
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
        source = core._normalize_source_path(indexed.rel_path)
        if not source:
            continue
        for kind, reference in core._iter_local_path_references_in_lines(indexed.iter_normalized_lines()):
            target = core._resolve_document_reference(
                source,
                reference,
                source_lookup,
                basename_lookup,
            )
            if target is None:
                continue
            core._add_graph_edge(edges_by_source, source, target, kind, reference)

    token_sources: dict[str, set[str]] = {}
    for indexed in indexed_files:
        source = core._normalize_source_path(indexed.rel_path)
        if not source:
            continue
        for token in core._extract_shared_tokens(indexed):
            token_sources.setdefault(token, set()).add(source)

    for token, matched_sources in token_sources.items():
        if not 2 <= len(matched_sources) <= core.GRAPH_SHARED_TOKEN_MAX_DOC_FREQ:
            continue
        ordered_sources = sorted(matched_sources)
        for source in ordered_sources:
            for target in ordered_sources:
                if source == target:
                    continue
                core._add_graph_edge(edges_by_source, source, target, "shared_symbol", token)

    siblings_by_dir: dict[str, list[str]] = {}
    for source in sources:
        siblings_by_dir.setdefault(core.posixpath.dirname(source), []).append(source)

    for siblings in siblings_by_dir.values():
        if len(siblings) < 2:
            continue
        normalized_stems = {
            source: core._stem_tokens(Path(source).stem)
            for source in siblings
        }
        ordered = sorted(siblings)
        for source in ordered:
            source_stem = Path(source).stem.lower()
            source_tokens = normalized_stems[source]
            for target in ordered:
                if source == target:
                    continue
                target_stem = Path(target).stem.lower()
                target_tokens = normalized_stems[target]
                shared_tokens = sorted(source_tokens & target_tokens)
                if source_stem in target_stem or target_stem in source_stem or len(shared_tokens) >= 2:
                    reason = ",".join(shared_tokens) or target_stem
                    core._add_graph_edge(edges_by_source, source, target, "same_series", reason)

    neighbors: dict[str, list[dict[str, Any]]] = {}
    edge_count = 0
    for source in sources:
        source_kb = core._extract_kb(source)
        ranked = sorted(
            edges_by_source[source].values(),
            key=lambda item: (
                item[0],
                0 if core._extract_kb(item[1]["target"]) == source_kb else 1,
                item[1]["target"],
            ),
        )
        selected = [payload for _, payload in ranked[: core.GRAPH_MAX_NEIGHBORS]]
        edge_count += len(selected)
        neighbors[source] = selected

    return {
        "version": 1,
        "edge_count": edge_count,
        "neighbors": neighbors,
    }


def _clear_generated_wiki_artifacts(persist_path: Path) -> None:
    core = _core()
    wiki_dir = persist_path / core.WIKI_DIRNAME
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
    core = _core()
    normalized_dir = persist_path / core.NORMALIZED_TEXT_DIRNAME
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

    skip_graph = core._env_flag("SKIP_GRAPH")
    skip_semantic = skip_graph or core._env_flag("SKIP_SEMANTIC")
    skip_wiki = core._env_flag("SKIP_WIKI")
    manifest = {
        "build_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "embed_model": embed_model,
        "total_chunks": total_chunks,
        "kb_enabled": True,
        "source_dir": str(Path(source_dir).expanduser()),
        "normalized_text_dir": core.NORMALIZED_TEXT_DIRNAME,
        "symbol_index_file": core.SYMBOL_INDEX_FILENAME,
        "document_graph_file": core.DOCUMENT_GRAPH_FILENAME,
        "entity_graph_file": core.ENTITY_GRAPH_FILENAME,
        "semantic_extract_cache_file": core.SEMANTIC_EXTRACT_CACHE_FILENAME,
        "community_index_file": core.COMMUNITY_INDEX_FILENAME,
        "graph_report_file": f"{core.REPORTS_DIRNAME}/{core.GRAPH_REPORT_FILENAME}",
        "lint_report_file": core.LINT_REPORT_FILENAME,
        "search_mode_default": os.getenv("SEARCH_MODE", core.DEFAULT_SEARCH_MODE).strip() or core.DEFAULT_SEARCH_MODE,
        "semantic_graph_stats": {},
        "build_flags": {
            "skip_graph": skip_graph,
            "skip_semantic": skip_semantic,
            "skip_wiki": skip_wiki,
        },
        "files": [],
    }

    symbol_index_path = persist_path / core.SYMBOL_INDEX_FILENAME
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

    file_sources = [core._normalize_source_path(indexed.rel_path) for indexed in indexed_files]
    if skip_graph:
        document_graph = {"version": 1, "edge_count": 0, "neighbors": {}}
    else:
        document_graph = core._build_document_graph(indexed_files)
    document_graph_path = persist_path / core.DOCUMENT_GRAPH_FILENAME
    document_graph_path.write_text(
        json.dumps(document_graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    semantic_sections: list[dict[str, Any]] = []
    if skip_semantic:
        core._write_semantic_cache(persist_path, {"version": 1, "entries": {}})
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
        semantic_sections, semantic_stats = core._extract_semantic_sections(
            indexed_files=indexed_files,
            persist_path=persist_path,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
        )
    manifest["semantic_graph_stats"] = semantic_stats

    if skip_graph:
        entity_graph = core._build_file_only_entity_graph(file_sources, document_graph)
    else:
        entity_graph = core._build_entity_graph(
            indexed_files,
            document_graph=document_graph,
            semantic_sections=semantic_sections,
        )
    entity_graph = core._merge_query_notes_into_entity_graph(
        entity_graph,
        core._load_query_note_records(persist_path),
    )
    entity_graph_path = persist_path / core.ENTITY_GRAPH_FILENAME
    entity_graph_path.write_text(
        json.dumps(entity_graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    community_index = core._build_community_index(
        file_sources=file_sources,
        document_graph=document_graph,
        entity_graph=entity_graph,
        semantic_stats=semantic_stats,
    )
    community_index_path = persist_path / core.COMMUNITY_INDEX_FILENAME
    community_index_path.write_text(
        json.dumps(community_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    reports_dir = persist_path / core.REPORTS_DIRNAME
    reports_dir.mkdir(parents=True, exist_ok=True)
    graph_report_path = reports_dir / core.GRAPH_REPORT_FILENAME
    graph_report_path.write_text(
        core._render_graph_report(community_index, manifest),
        encoding="utf-8",
    )

    (persist_path / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


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
    core = _core()
    for attempt in range(max_retries + 1):
        try:
            return embeddings_factory().embed_documents(texts)
        except Exception as exc:
            if not core._is_rate_limit_error(exc) or attempt >= max_retries:
                raise
            delay = retry_base_seconds * (2**attempt)
            if rate_limit_callback is not None:
                rate_limit_callback(delay, attempt + 1, max_retries)
            core.time.sleep(delay)
    raise RuntimeError("embedding 批处理在重试后仍未成功。")


def _add_embedded_batch(
    *,
    batch: list[Document],
    embedded_texts: list[list[float]],
    embeddings: OpenAIEmbeddings,
    vectorstore: FAISS | None,
) -> FAISS:
    core = _core()
    texts = [document.page_content for document in batch]
    metadatas = [document.metadata for document in batch]
    ids = _document_batch_ids(batch)
    text_embeddings = zip(texts, embedded_texts)
    if vectorstore is None:
        if not hasattr(core.FAISS, "from_embeddings"):
            return core.FAISS.from_documents(batch, embeddings)
        return core.FAISS.from_embeddings(
            text_embeddings,
            embeddings,
            metadatas=metadatas,
            ids=ids,
        )
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
    core = _core()

    def _cb(current: int, total: int, message: str) -> None:
        if progress_callback:
            progress_callback(current, total, message)

    if total_chunks == 0:
        raise ValueError(
            f"在 {md_dir} 中未找到可索引文本。支持后缀: {', '.join(sorted(core.SUPPORTED_TEXT_SUFFIXES))}"
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
    embeddings = core.make_embeddings(api_key=embed_api_key, base_url=embed_base_url, model=embed_model)

    batch_size = max(1, int(os.getenv("EMBED_BATCH_SIZE", str(core.DEFAULT_EMBED_BATCH_SIZE))))
    batch_sleep_seconds = max(
        0.0,
        float(os.getenv("EMBED_BATCH_SLEEP_SECONDS", str(core.DEFAULT_EMBED_BATCH_SLEEP_SECONDS))),
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
                core.time.sleep(batch_sleep_seconds)
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
                client = core.make_embeddings(
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

    manifest = core._write_index_artifacts(
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
        core._clear_generated_wiki_artifacts(persist_path)
        _cb(total_chunks + 2, total_steps, "已按配置跳过离线 wiki 生成。")
    else:
        _cb(total_chunks + 2, total_steps, "正在生成离线 wiki 导航...")
        from wiki import generate_wiki

        generate_wiki(persist_path=persist_path, manifest=manifest)
    core._read_cached_text.cache_clear()
    _cb(total_chunks + 3, total_steps, "正在加载检索 bundle...")

    bundle = core.load_search_bundle(
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
    embed_base_url: str = "https://api.siliconflow.cn/v1",
    embed_model: str = "BAAI/bge-m3",
    persist_dir: str = str(Path.home() / "wechat_rag_db"),
    progress_callback: Callable[[int, int, str], None] | None = None,
    llm_api_key: str = "",
    llm_model: str = "",
    llm_base_url: str = "",
) -> SearchBundle:
    core = _core()

    def _cb(current: int, total: int, message: str) -> None:
        if progress_callback:
            progress_callback(current, total, message)

    _cb(0, 1, "正在扫描目录并解析文本文件...")
    if core._build_documents is not _ORIGINAL_BUILD_DOCUMENTS:
        documents, indexed_files = core._build_documents(md_dir)
        return core._build_vectorstore_from_document_stream(
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
        cache_root = Path(tmpdir) / core.NORMALIZED_TEXT_DIRNAME
        indexed_files, total_chunks = core._index_source_files(md_dir, cache_root)
        return core._build_vectorstore_from_document_stream(
            indexed_files=indexed_files,
            documents=core._iter_documents_for_indexed_files(indexed_files),
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
