from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Any, Iterable


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


_MARKDOWN_FENCE_RE = re.compile(r"^\s*([`~]{3,})")
_MARKDOWN_LIST_ITEM_RE = re.compile(r"^\s{0,3}(?:[-+*]|\d+[.)])\s+")
_MARKDOWN_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
_MARKDOWN_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$"
)
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?")
_MEETING_TIMESTAMP_RE = re.compile(
    r"^\s*(?:\[\d{1,2}:\d{2}(?::\d{2})?\]|\d{1,2}:\d{2}(?::\d{2})?)\s+"
    r"(?:(?:[\u4e00-\u9fffA-Za-z][^:：]{0,20})[:：]|[-–—])"
)
_MEETING_SPEAKER_RE = re.compile(
    r"^\s*(?:[\u4e00-\u9fff]{2,12}|[A-Z][A-Za-z0-9_. -]{1,30})\s*[:：]\s+\S+"
)
_LOG_LEVEL_PREFIXES = {
    "TRACE",
    "DEBUG",
    "INFO",
    "WARN",
    "WARNING",
    "ERROR",
    "FATAL",
    "CRITICAL",
}
_METADATA_HEADER_KEYS = {
    "title",
    "author",
    "version",
    "date",
    "time",
    "created",
    "updated",
    "status",
    "summary",
    "description",
    "subject",
    "owner",
    "category",
    "tags",
}


def _looks_like_log_prefix(line: str) -> bool:
    match = re.match(r"^\s*([A-Za-z][A-Za-z0-9_]{1,20})\s*[:：]\s+\S+", line)
    if not match:
        return False
    return match.group(1).upper() in _LOG_LEVEL_PREFIXES


def _looks_like_metadata_header(line: str) -> bool:
    match = re.match(r"^\s*([A-Za-z][A-Za-z0-9_ -]{1,30})\s*[:：]\s+\S+", line)
    if not match:
        return False
    normalized = re.sub(r"[\s_-]+", " ", match.group(1).strip().lower())
    return normalized in _METADATA_HEADER_KEYS


def _looks_like_meeting_boundary(line: str) -> bool:
    if _looks_like_log_prefix(line) or _looks_like_metadata_header(line):
        return False
    return bool(_MEETING_TIMESTAMP_RE.match(line) or _MEETING_SPEAKER_RE.match(line))


def _looks_like_meeting_notes(text: str) -> bool:
    timestamp_matches = 0
    speaker_matches = 0
    scanned = 0
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        scanned += 1
        if _MEETING_TIMESTAMP_RE.match(stripped):
            timestamp_matches += 1
        elif _MEETING_SPEAKER_RE.match(stripped) and not (
            _looks_like_log_prefix(stripped) or _looks_like_metadata_header(stripped)
        ):
            speaker_matches += 1
        if scanned >= 80:
            break
    return timestamp_matches >= 2 or speaker_matches >= 3


def _classify_structured_line(line: str, *, markdown: bool) -> str:
    if markdown and _MARKDOWN_TABLE_RE.match(line):
        return "table"
    if markdown and _BLOCKQUOTE_RE.match(line):
        return "quote"
    if markdown and _MARKDOWN_LIST_ITEM_RE.match(line):
        return "list"
    return "paragraph"


def _structured_block_label(
    kind: str,
    *,
    default_label: str,
    first_line: str,
) -> str:
    stripped = first_line.strip()
    if kind == "meeting":
        return stripped[:48]
    if kind == "code":
        return "code block"
    if kind == "table":
        return "table"
    if kind == "list":
        return default_label or "list"
    if kind == "quote":
        return default_label or "quote"
    return default_label


def _emit_structured_block(
    lines: list[str],
    start_line: int | None,
    *,
    end_line: int | None,
    label: str,
    section_path: tuple[str, ...] = (),
    kind: str = "",
) -> Any:
    core = _core()
    if not lines or start_line is None or end_line is None:
        return None
    text = "\n".join(lines).strip()
    if not text:
        return None
    return core.ChunkSpec(
        text=text,
        line_start=start_line,
        line_end=end_line,
        label=label,
        section_path=section_path,
        kind=kind,
    )


def _iter_structured_blocks_from_line_stream(
    numbered_lines: Iterable[tuple[int, str]],
    *,
    markdown: bool = False,
    detect_meeting: bool = False,
    default_label: str = "chunk",
    section_path: tuple[str, ...] = (),
) -> Iterable[Any]:
    current_lines: list[str] = []
    current_start: int | None = None
    current_end: int | None = None
    current_kind = ""
    current_label = default_label
    in_code_fence = False
    active_fence = ""

    def flush() -> Iterable[Any]:
        nonlocal current_lines, current_start, current_end, current_kind, current_label
        block = _emit_structured_block(
            current_lines,
            current_start,
            end_line=current_end,
            label=current_label,
            section_path=section_path,
            kind=current_kind,
        )
        current_lines = []
        current_start = None
        current_end = None
        current_kind = ""
        current_label = default_label
        if block is not None:
            yield block

    def begin_block(kind: str, line_no: int, line: str) -> None:
        nonlocal current_lines, current_start, current_end, current_kind, current_label
        current_lines = [line]
        current_start = line_no
        current_end = line_no
        current_kind = kind
        current_label = _structured_block_label(kind, default_label=default_label, first_line=line)

    for line_no, line in numbered_lines:
        stripped = line.strip()
        fence_match = _MARKDOWN_FENCE_RE.match(line) if markdown else None

        if in_code_fence:
            current_lines.append(line)
            current_end = line_no
            if fence_match and fence_match.group(1)[0] == active_fence[0] and len(fence_match.group(1)) >= len(active_fence):
                in_code_fence = False
                active_fence = ""
                yield from flush()
            continue

        if not stripped:
            if current_lines:
                yield from flush()
            continue

        if markdown and fence_match:
            if current_lines:
                yield from flush()
            begin_block("code", line_no, line)
            in_code_fence = True
            active_fence = fence_match.group(1)
            continue

        if detect_meeting and _looks_like_meeting_boundary(stripped):
            if current_lines:
                yield from flush()
            begin_block("meeting", line_no, line)
            continue

        line_kind = _classify_structured_line(line, markdown=markdown)
        if not current_lines:
            begin_block(line_kind, line_no, line)
            continue

        should_continue = False
        if current_kind == "meeting":
            should_continue = True
        elif current_kind == "paragraph":
            should_continue = line_kind == "paragraph"
        elif current_kind == "list":
            should_continue = line_kind == "list" or line.startswith((" ", "\t"))
        elif current_kind == "table":
            should_continue = line_kind == "table" or _MARKDOWN_TABLE_SEPARATOR_RE.match(line) is not None
        elif current_kind == "quote":
            should_continue = line_kind == "quote"

        if should_continue:
            current_lines.append(line)
            current_end = line_no
            continue

        yield from flush()
        begin_block(line_kind, line_no, line)

    if current_lines:
        yield from flush()


def _pack_structured_blocks(
    blocks: Iterable[Any],
    *,
    chunk_size: int | None = None,
    overlap_lines: int | None = None,
    default_label: str = "chunk",
    section_path: tuple[str, ...] = (),
) -> list[Any]:
    core = _core()
    chunk_size = core._configured_chunk_size() if chunk_size is None else max(1, chunk_size)
    overlap_lines = (
        core._configured_chunk_overlap_lines()
        if overlap_lines is None
        else max(0, overlap_lines)
    )
    results: list[Any] = []
    current_blocks: list[Any] = []
    current_len = 0
    chunk_index = 0

    def block_line_count(block: Any) -> int:
        line_start = getattr(block, "line_start", None)
        line_end = getattr(block, "line_end", None)
        if line_start is None or line_end is None:
            return max(1, len(str(getattr(block, "text", "") or "").splitlines()))
        return max(1, int(line_end) - int(line_start) + 1)

    def oversized_block(block: Any) -> list[Any]:
        return core._split_lines_into_chunks(
            str(getattr(block, "text", "") or "").splitlines(),
            start_line=int(getattr(block, "line_start", 1) or 1),
            chunk_size=chunk_size,
            overlap_lines=overlap_lines,
            label=str(getattr(block, "label", "") or default_label),
            section_path=getattr(block, "section_path", ()) or section_path,
            kind=str(getattr(block, "kind", "") or ""),
        )

    def emit_current() -> None:
        nonlocal current_blocks, current_len, chunk_index
        if not current_blocks:
            return
        chunk_index += 1
        text = "\n\n".join(str(block.text).strip() for block in current_blocks if str(block.text).strip()).strip()
        if not text:
            current_blocks = []
            current_len = 0
            return
        effective_section_path = section_path or (current_blocks[0].section_path if current_blocks else ())
        specific_label = next(
            (
                str(block.label).strip()
                for block in current_blocks
                if str(block.label).strip() and str(block.label).strip() not in {"chunk", "preface"}
            ),
            "",
        )
        if default_label and default_label != "chunk":
            label = default_label
        else:
            label = specific_label or (effective_section_path[-1] if effective_section_path else f"chunk {chunk_index}")
        first_kind = current_blocks[0].kind if current_blocks else ""
        chunk_kind = first_kind if all(block.kind == first_kind for block in current_blocks) else "mixed"
        results.append(
            core.ChunkSpec(
                text=text,
                line_start=current_blocks[0].line_start,
                line_end=current_blocks[-1].line_end,
                label=label,
                section_path=effective_section_path,
                kind=chunk_kind,
            )
        )
        if overlap_lines > 0:
            retained: list[Any] = []
            retained_lines = 0
            for block in reversed(current_blocks):
                retained.insert(0, block)
                retained_lines += block_line_count(block)
                if retained_lines >= overlap_lines:
                    break
            current_blocks = retained
            current_len = sum(len(str(block.text)) + 2 for block in current_blocks)
        else:
            current_blocks = []
            current_len = 0

    for block in blocks:
        block_text = str(getattr(block, "text", "") or "")
        if not block_text.strip():
            continue
        if len(block_text) > chunk_size:
            emit_current()
            split_blocks = oversized_block(block)
            if split_blocks:
                results.extend(split_blocks)
            current_blocks = []
            current_len = 0
            continue

        block_len = len(block_text) + 2
        if current_blocks and current_len + block_len > chunk_size:
            emit_current()
        current_blocks.append(block)
        current_len += block_len

    emit_current()
    return results


def _chunk_markdown_by_structure(text: str, default_label: str = "preface") -> list[Any]:
    lines = text.splitlines()
    return _pack_structured_blocks(
        _iter_structured_blocks_from_line_stream(
            enumerate(lines, start=1),
            markdown=True,
            default_label=default_label,
            section_path=(default_label,) if default_label and default_label != "chunk" else (),
        ),
        default_label=default_label,
        section_path=(default_label,) if default_label and default_label != "chunk" else (),
    )


def _chunk_plain_text_by_structure(text: str, default_label: str = "chunk") -> list[Any]:
    detect_meeting = _looks_like_meeting_notes(text)
    lines = text.splitlines()
    return _pack_structured_blocks(
        _iter_structured_blocks_from_line_stream(
            enumerate(lines, start=1),
            markdown=False,
            detect_meeting=detect_meeting,
            default_label=default_label,
        ),
        default_label=default_label,
    )


def _chunk_structured_text(normalized_text: str) -> list[Any]:
    core = _core()
    return core._split_lines_into_chunks(normalized_text.splitlines())


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
                section_path=(label,),
                kind="section",
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
                    section_path=("preface",),
                    kind="section",
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
    section_path: tuple[str, ...] = (),
    kind: str = "",
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
        section_path=section_path,
        kind=kind,
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
    segment_section_path: tuple[str, ...] = ("preface",)
    current_len = 0
    saw_heading = False

    def push_line(line_no: int, line: str) -> Iterable[Any]:
        nonlocal segment_lines, segment_start_line, current_len
        line_len = len(line) + 1
        if segment_lines and current_len + line_len > chunk_size:
            chunk = _emit_line_chunk(
                segment_lines,
                segment_start_line,
                label=segment_label,
                section_path=segment_section_path,
                kind="section",
            )
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
        chunk = _emit_line_chunk(
            segment_lines,
            segment_start_line,
            label=segment_label,
            section_path=segment_section_path,
            kind="section",
        )
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
            segment_section_path = (segment_label,)
            yield from push_line(line_no, line)
            continue

        if not saw_heading:
            segment_label = "preface"
            segment_section_path = ("preface",)
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
    if chunk_strategy == "markdown_structure":
        return iter(
            core._limit_chunk_specs(
                _pack_structured_blocks(
                    _iter_structured_blocks_from_line_stream(
                        enumerate(core._iter_text_file_lines(path), start=1),
                        markdown=True,
                        default_label="preface",
                        section_path=("preface",),
                    ),
                    default_label="preface",
                    section_path=("preface",),
                ),
                chunk_strategy=chunk_strategy,
            )
        )
    if chunk_strategy in {"meeting_notes", "plain_text_structure"}:
        return iter(
            core._limit_chunk_specs(
                _pack_structured_blocks(
                    _iter_structured_blocks_from_line_stream(
                        enumerate(core._iter_text_file_lines(path), start=1),
                        markdown=False,
                        detect_meeting=chunk_strategy == "meeting_notes",
                    )
                ),
                chunk_strategy=chunk_strategy,
            )
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
    body_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not body_nodes:
        return core._split_lines_into_chunks(lines)

    chunks: list[Any] = []
    preface_end = body_nodes[0].lineno - 1
    if preface_end > 0:
        chunks.extend(
            core._split_lines_into_chunks(
                lines[:preface_end],
                start_line=1,
                label="module prelude",
                section_path=("module prelude",),
                kind="code",
            )
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
                section_path=(label,),
                kind="code",
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
