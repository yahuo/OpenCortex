from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import re
import shutil
from typing import Any

WIKI_DIRNAME = "wiki"
QUERY_NOTES_DIRNAME = "queries"
NORMALIZED_TEXT_DIRNAME = "normalized_texts"
MAX_PREVIEW_LINES = 12
MAX_PREVIEW_CHARS = 1200
LINK_TEXT_ESCAPES = str.maketrans({
    "\\": "\\\\",
    "[": "\\[",
    "]": "\\]",
})
LINK_DEST_ESCAPES = str.maketrans({
    "%": "%25",
    " ": "%20",
    "#": "%23",
    "?": "%3F",
    "(": "%28",
    ")": "%29",
    "<": "%3C",
    ">": "%3E",
})


def _iter_file_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    files = manifest.get("files", [])
    valid_entries: list[dict[str, Any]] = []
    if not isinstance(files, list):
        return valid_entries

    for entry in files:
        if not isinstance(entry, dict):
            continue
        source = entry.get("name", "")
        normalized_text = entry.get("normalized_text", "")
        if not isinstance(source, str) or not source:
            continue
        if not isinstance(normalized_text, str) or not normalized_text:
            continue
        valid_entries.append(entry)

    return sorted(valid_entries, key=lambda item: (str(item.get("kb", "")), item["name"]))


def _page_relpath_for_source(source: str) -> Path:
    return Path("files") / Path(f"{source}.md")


def _escape_markdown_link_text(text: str) -> str:
    return text.translate(LINK_TEXT_ESCAPES)


def _encode_markdown_link_destination(target: str) -> str:
    return target.translate(LINK_DEST_ESCAPES)


def _markdown_link(label: str, target: str) -> str:
    return f"[{_escape_markdown_link_text(label)}]({_encode_markdown_link_destination(target)})"


def _longest_fence_run(text: str, marker: str) -> int:
    runs = re.findall(re.escape(marker) + r"+", text)
    return max((len(run) for run in runs), default=0)


def _wrap_preview_block(preview: str) -> list[str]:
    backtick_len = max(3, _longest_fence_run(preview, "`") + 1)
    tilde_len = max(3, _longest_fence_run(preview, "~") + 1)
    marker, fence_len = ("~", tilde_len) if tilde_len < backtick_len else ("`", backtick_len)
    fence = marker * fence_len
    return [f"{fence}text", preview, fence]


def _inline_code(text: str) -> str:
    fence_len = max(1, _longest_fence_run(text, "`") + 1)
    fence = "`" * fence_len
    padded = text
    if text.startswith(("`", " ")) or text.endswith(("`", " ")):
        padded = f" {text} "
    return f"{fence}{padded}{fence}"


def _slugify_query(text: str, max_len: int = 64) -> str:
    slug = re.sub(r"\s+", "-", text.strip().lower())
    slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", slug, flags=re.UNICODE)
    slug = re.sub(r"-{2,}", "-", slug).strip("-_")
    if not slug:
        return "query-note"
    return slug[:max_len].rstrip("-_") or "query-note"


def _format_source_location(source: dict[str, Any]) -> str:
    time_range = str(source.get("time_range", "") or "").strip()
    if time_range:
        return time_range
    line_start = source.get("line_start")
    line_end = source.get("line_end")
    if line_start:
        if line_end and line_end != line_start:
            return f"L{line_start}-L{line_end}"
        return f"L{line_start}"
    return "unknown"


def _query_note_relpath(source_relpath: Path) -> Path:
    return Path(QUERY_NOTES_DIRNAME) / source_relpath


def _next_query_note_path(queries_dir: Path, note_date: str, slug: str) -> tuple[Path, Path]:
    base_name = f"{note_date}-{slug}"
    candidate_rel = Path(f"{base_name}.md")
    candidate_path = queries_dir / candidate_rel
    suffix = 2
    while candidate_path.exists():
        candidate_rel = Path(f"{base_name}-{suffix}.md")
        candidate_path = queries_dir / candidate_rel
        suffix += 1
    return candidate_path, _query_note_relpath(candidate_rel)


def _render_query_note(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    created_at: datetime,
    note_relpath: Path,
    tags: list[str] | None = None,
) -> str:
    lines = [
        f"# {question}",
        "",
        f"- 生成时间：{_inline_code(created_at.isoformat(timespec='seconds'))}",
        f"- 类型：{_inline_code('query_note')}",
        f"- 路径：{_inline_code(note_relpath.as_posix())}",
    ]

    clean_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
    if clean_tags:
        lines.append(f"- 标签：{', '.join(_inline_code(tag) for tag in clean_tags)}")

    lines.extend(
        [
            "",
            "## 问题",
            "",
            question,
            "",
            "## 结论",
            "",
            answer.strip(),
            "",
            "## 来源",
            "",
        ]
    )

    for index, source in enumerate(sources, start=1):
        source_name = str(source.get("source", "") or "unknown")
        page_rel = Path("..") / _page_relpath_for_source(source_name)
        lines.extend(
            [
                f"### 证据 {index}",
                "",
                f"- 来源文件：{_markdown_link(source_name, page_rel.as_posix())}",
                f"- 位置：{_inline_code(_format_source_location(source))}",
                f"- 匹配类型：{_inline_code(str(source.get('match_kind', 'source') or 'source'))}",
            ]
        )
        snippet = str(source.get("snippet", "") or "").strip()
        if snippet:
            lines.extend(["- 摘要：", ""])
            lines.extend(_wrap_preview_block(snippet))
        lines.append("")

    lines.extend(
        [
            "## 备注",
            "",
            "此条知识由显式“保存为知识”操作生成，默认不进入最高优先级检索证据链。",
            "",
        ]
    )
    return "\n".join(lines)


def _append_query_note_log(log_path: Path, created_at: datetime, question: str, note_relpath: Path) -> None:
    entry = f"- {created_at.isoformat(timespec='seconds')} {_markdown_link(question, note_relpath.as_posix())}"
    if log_path.exists():
        content = log_path.read_text(encoding="utf-8").rstrip()
    else:
        content = "\n".join(
            [
                "# 知识日志",
                "",
                "此文件记录构建阶段产物和显式保存的 query notes。",
            ]
        ).rstrip()

    if "## Query Notes" not in content:
        content = content + "\n\n## Query Notes\n"
    if entry not in content:
        content = content.rstrip() + "\n" + entry + "\n"
    else:
        content = content.rstrip() + "\n"
    log_path.write_text(content, encoding="utf-8")


def _preview_from_normalized_text(normalized_root: Path, normalized_rel: str) -> str:
    normalized_path = normalized_root / normalized_rel
    if not normalized_path.exists():
        return ""

    text = normalized_path.read_text(encoding="utf-8")
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    preview = "\n".join(lines[:MAX_PREVIEW_LINES]).strip()
    if len(preview) > MAX_PREVIEW_CHARS:
        preview = preview[: MAX_PREVIEW_CHARS - 3].rstrip() + "..."
    return preview


def _render_index(entries: list[dict[str, Any]], build_time: str) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        kb = str(entry.get("kb", "") or "未分类")
        grouped[kb].append(entry)

    lines = [
        "# 知识库导航",
        "",
        "此目录由构建阶段自动生成，当前只用于人类导航，不参与运行时检索。",
        "",
        f"- 最近构建：`{build_time or 'unknown'}`",
        f"- 文件总数：`{len(entries)}`",
        f"- 知识库数量：`{len(grouped)}`",
        "",
    ]

    for kb, group in sorted(grouped.items()):
        lines.extend(
            [
                f"## {kb}",
                "",
            ]
        )
        for entry in group:
            source = str(entry["name"])
            suffix = str(entry.get("suffix", "") or "unknown")
            chunks = int(entry.get("chunks") or 0)
            page_rel = _page_relpath_for_source(source).as_posix()
            lines.append(
                f"- {_markdown_link(source, page_rel)} · `{suffix}` · `{chunks}` chunks"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_file_page(entry: dict[str, Any], preview: str) -> str:
    source = str(entry["name"])
    kb = str(entry.get("kb", "") or "未分类")
    suffix = str(entry.get("suffix", "") or "unknown")
    size_kb = entry.get("size_kb", "unknown")
    mtime = str(entry.get("mtime", "") or "unknown")
    chunks = int(entry.get("chunks") or 0)
    normalized_text = str(entry.get("normalized_text", ""))

    lines = [
        f"# {source}",
        "",
        f"- 知识库：{_inline_code(kb)}",
        f"- 类型：{_inline_code(suffix)}",
        f"- 大小：{_inline_code(f'{size_kb} KB')}",
        f"- 更新时间：{_inline_code(mtime)}",
        f"- 分片数：{_inline_code(str(chunks))}",
        f"- 标准化文本：{_inline_code(normalized_text)}",
        "",
        "## 预览",
        "",
    ]

    if preview:
        lines.extend(_wrap_preview_block(preview))
    else:
        lines.append("_无可用预览_")

    lines.extend(
        [
            "",
            "## 备注",
            "",
            "此页面由构建阶段自动生成，当前仅用于人类导航。",
            "",
        ]
    )
    return "\n".join(lines)


def _render_log(entries: list[dict[str, Any]], build_time: str) -> str:
    return "\n".join(
        [
            "# 知识日志",
            "",
            "此文件由构建阶段自动创建，当前作为后续知识沉淀的占位页。",
            "",
            f"- 最近构建：`{build_time or 'unknown'}`",
            f"- 文件总数：`{len(entries)}`",
            "",
        ]
    )


def generate_wiki(persist_path: Path, manifest: dict[str, Any]) -> dict[str, int]:
    entries = _iter_file_entries(manifest)
    build_time = str(manifest.get("build_time", "") or "unknown")
    normalized_root = persist_path / str(
        manifest.get("normalized_text_dir", NORMALIZED_TEXT_DIRNAME) or NORMALIZED_TEXT_DIRNAME
    )

    wiki_dir = persist_path / WIKI_DIRNAME
    files_dir = wiki_dir / "files"
    if files_dir.exists():
        shutil.rmtree(files_dir)
    files_dir.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        source = str(entry["name"])
        page_path = wiki_dir / _page_relpath_for_source(source)
        page_path.parent.mkdir(parents=True, exist_ok=True)
        preview = _preview_from_normalized_text(normalized_root, str(entry["normalized_text"]))
        page_path.write_text(_render_file_page(entry, preview), encoding="utf-8")

    wiki_dir.mkdir(parents=True, exist_ok=True)
    (wiki_dir / "index.md").write_text(_render_index(entries, build_time), encoding="utf-8")
    (wiki_dir / "log.md").write_text(_render_log(entries, build_time), encoding="utf-8")
    return {"pages": len(entries)}


def save_query_note(
    persist_path: Path,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    tags: list[str] | None = None,
    created_at: datetime | None = None,
) -> dict[str, str]:
    question = question.strip()
    answer = answer.strip()
    valid_sources = [item for item in sources if isinstance(item, dict) and str(item.get("source", "")).strip()]
    if not question:
        raise ValueError("问题不能为空")
    if not answer:
        raise ValueError("结论不能为空")
    if not valid_sources:
        raise ValueError("保存 query note 时必须包含至少一条来源")

    created = created_at or datetime.now().astimezone()
    wiki_dir = persist_path / WIKI_DIRNAME
    queries_dir = wiki_dir / QUERY_NOTES_DIRNAME
    wiki_dir.mkdir(parents=True, exist_ok=True)
    queries_dir.mkdir(parents=True, exist_ok=True)

    note_path, note_relpath = _next_query_note_path(
        queries_dir,
        note_date=created.date().isoformat(),
        slug=_slugify_query(question),
    )
    note_path.write_text(
        _render_query_note(question, answer, valid_sources, created, note_relpath, tags=tags),
        encoding="utf-8",
    )

    log_path = wiki_dir / "log.md"
    _append_query_note_log(log_path, created, question, note_relpath)
    return {
        "note_path": str(note_path),
        "note_relpath": (Path(WIKI_DIRNAME) / note_relpath).as_posix(),
        "log_path": str(log_path),
    }
