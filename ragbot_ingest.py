from __future__ import annotations

import configparser
import json
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragbot_chunking import (
    _chunk_markdown_by_structure,
    _chunk_plain_text_by_structure,
    _looks_like_meeting_notes,
)

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
    from ragbot import IndexedFile


def _core():
    import ragbot as core

    return core


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


def _convert_binary_to_markdown(file_path: Path) -> str:
    if not _MARKITDOWN_AVAILABLE:
        raise ImportError(
            f"处理 {file_path.suffix} 文件需要 markitdown 库，"
            "请运行: pip install 'markitdown[pdf,docx,xlsx]'"
        )
    md = MarkItDown()
    result = md.convert(str(file_path))
    return result.text_content


def _wechat_chunk_specs(messages: list[dict[str, Any]]) -> list[Any]:
    core = _core()
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
    return core._ensure_chunk_size(chunk_specs)


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
            chunks = _wechat_chunk_specs(messages)
        else:
            if any(core.re.match(r"^\s{0,3}#{1,6}\s+", line) for line in normalized_raw.splitlines()):
                chunk_strategy = "markdown_heading"
                chunks = core._chunk_markdown_by_heading(normalized_raw)
            else:
                chunk_strategy = "markdown_structure"
                chunks = _chunk_markdown_by_structure(normalized_raw)
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
        if _looks_like_meeting_notes(normalized_text):
            chunk_strategy = "meeting_notes"
        else:
            chunk_strategy = "plain_text_structure"
        chunks = _chunk_plain_text_by_structure(normalized_text)

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
    if strategy == "wechat_markdown" or (
        (suffix in core.MARKDOWN_SUFFIXES or suffix in core._BINARY_SUFFIXES) and not strategy
    ):
        messages = core._parse_md_messages_text(normalized_text) if suffix in core.MARKDOWN_SUFFIXES else []
        if messages:
            chunks = _wechat_chunk_specs(messages)
        else:
            if any(core.re.match(r"^\s{0,3}#{1,6}\s+", line) for line in normalized_text.splitlines()):
                chunks = core._chunk_markdown_by_heading(normalized_text)
            else:
                chunks = _chunk_markdown_by_structure(normalized_text)
    elif strategy == "markdown_heading":
        chunks = core._chunk_markdown_by_heading(normalized_text)
    elif strategy == "markdown_structure":
        chunks = _chunk_markdown_by_structure(normalized_text)
    elif strategy == "structured" or suffix in core.STRUCTURED_SUFFIXES:
        chunks = core._chunk_structured_text(normalized_text)
    elif strategy == "python" or suffix in core.PYTHON_SUFFIXES:
        chunks = core._chunk_python_code(normalized_text)
    elif strategy in {"meeting_notes", "plain_text_structure"}:
        chunks = _chunk_plain_text_by_structure(normalized_text)
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

    normalized_text, chunks, symbols, chunk_strategy, line_total, original_chunk_total = (
        core._prepare_indexed_content(
            rel_path,
            suffix,
            normalized_raw,
        )
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

    normalized_text, chunks, symbols, chunk_strategy, line_total, original_chunk_total = (
        core._prepare_indexed_content(
            rel_path,
            suffix,
            normalized_raw,
        )
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
