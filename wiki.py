from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any
from urllib.parse import unquote

try:
    import fcntl

    _FCNTL_AVAILABLE = True
except ImportError:
    fcntl = None
    _FCNTL_AVAILABLE = False

WIKI_DIRNAME = "wiki"
QUERY_NOTES_DIRNAME = "queries"
NORMALIZED_TEXT_DIRNAME = "normalized_texts"
LINT_REPORT_FILENAME = "lint_report.json"
WRITE_LOCK_FILENAME = ".write.lock"
WRITE_LOCK_TIMEOUT_SECONDS = 5.0
WRITE_LOCK_POLL_SECONDS = 0.05
STALE_LOCK_GRACE_SECONDS = 1.0
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
    content = _load_log_content(log_path)
    content = _merge_query_note_log_entry(content, entry)
    _write_text_atomically(log_path, content)


def _extract_query_notes_section(log_text: str) -> str:
    marker = "## Query Notes"
    if marker not in log_text:
        return ""
    _, _, tail = log_text.partition(marker)
    section = f"{marker}{tail}".strip()
    return section


def _load_log_content(log_path: Path) -> str:
    if log_path.exists():
        return log_path.read_text(encoding="utf-8").rstrip()
    return "\n".join(
        [
            "# 知识日志",
            "",
            "此文件记录构建阶段产物和显式保存的 query notes。",
        ]
    ).rstrip()


def _merge_query_note_log_entry(content: str, entry: str) -> str:
    if "## Query Notes" not in content:
        content = content + "\n\n## Query Notes\n"
    if entry not in content:
        content = content.rstrip() + "\n" + entry + "\n"
    else:
        content = content.rstrip() + "\n"
    return content


def _write_text_atomically(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _load_manifest(persist_path: Path) -> dict[str, Any] | None:
    manifest_path = persist_path / "index_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _expected_source_page_paths(wiki_dir: Path, sources: list[dict[str, Any]]) -> list[Path]:
    page_paths: list[Path] = []
    seen: set[str] = set()
    for item in sources:
        source = str(item.get("source", "") or "").strip()
        if not source or source in seen:
            continue
        seen.add(source)
        page_paths.append(wiki_dir / _page_relpath_for_source(source))
    return page_paths


def _ensure_wiki_artifacts_for_sources(
    persist_path: Path,
    manifest: dict[str, Any] | None,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("当前索引缺少 manifest，请先重建索引后再保存为知识。")

    manifest_sources = {
        str(entry.get("name", "")).strip()
        for entry in manifest.get("files", [])
        if isinstance(entry, dict) and str(entry.get("name", "")).strip()
    }
    requested_sources = {
        str(item.get("source", "")).strip()
        for item in sources
        if isinstance(item, dict) and str(item.get("source", "")).strip()
    }
    missing_sources = sorted(source for source in requested_sources if source not in manifest_sources)
    if missing_sources:
        raise ValueError(
            "当前索引不包含这些来源文件，无法保存为知识："
            + ", ".join(missing_sources)
        )

    wiki_dir = persist_path / WIKI_DIRNAME
    expected_paths = _expected_source_page_paths(wiki_dir, sources)
    index_path = wiki_dir / "index.md"
    log_path = wiki_dir / "log.md"
    missing_paths = [path for path in [index_path, log_path, *expected_paths] if not path.exists()]
    if missing_paths:
        _generate_wiki_unlocked(persist_path=persist_path, manifest=manifest)
    still_missing = [path for path in [index_path, log_path, *expected_paths] if not path.exists()]
    if still_missing:
        raise ValueError("当前索引还没有生成完整的 wiki 页面，请先重建索引后再保存为知识。")
    return manifest


@contextmanager
def _write_lock(wiki_dir: Path):
    lock_path = wiki_dir / WRITE_LOCK_FILENAME
    wiki_dir.mkdir(parents=True, exist_ok=True)
    if _FCNTL_AVAILABLE:
        with _posix_write_lock(lock_path):
            yield
        return

    with _portable_write_lock(lock_path):
        yield


@contextmanager
def _posix_write_lock(lock_path: Path):
    start = time.monotonic()
    lock_path.touch(exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.ftruncate(fd, 0)
                os.write(fd, f"{os.getpid()} {time.time_ns()}\n".encode("utf-8"))
                os.fsync(fd)
                break
            except BlockingIOError:
                if time.monotonic() - start >= WRITE_LOCK_TIMEOUT_SECONDS:
                    raise TimeoutError("等待 wiki 写锁超时，请稍后重试。")
                time.sleep(WRITE_LOCK_POLL_SECONDS)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _lock_owner_alive(lock_path: Path) -> bool:
    try:
        content = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    if not content:
        return False
    pid_text = content.split()[0]
    if not pid_text.isdigit():
        return False
    pid = int(pid_text)
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_is_stale(lock_path: Path) -> bool:
    if not lock_path.exists():
        return False
    if _lock_owner_alive(lock_path):
        return False
    try:
        age_seconds = time.time() - lock_path.stat().st_mtime
    except OSError:
        return False
    return age_seconds >= STALE_LOCK_GRACE_SECONDS


@contextmanager
def _portable_write_lock(lock_path: Path):
    start = time.monotonic()
    fd: int | None = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time_ns()}\n".encode("utf-8"))
            break
        except FileExistsError:
            if _lock_is_stale(lock_path):
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() - start >= WRITE_LOCK_TIMEOUT_SECONDS:
                raise TimeoutError("等待 wiki 写锁超时，请稍后重试。")
            time.sleep(WRITE_LOCK_POLL_SECONDS)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


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


def _render_log(entries: list[dict[str, Any]], build_time: str, existing_log_text: str = "") -> str:
    lines = [
        "# 知识日志",
        "",
        "此文件由构建阶段自动创建，用于记录构建产物和显式保存的 query notes。",
        "",
        f"- 最近构建：`{build_time or 'unknown'}`",
        f"- 文件总数：`{len(entries)}`",
        "",
    ]
    query_notes_section = _extract_query_notes_section(existing_log_text)
    if query_notes_section:
        lines.extend([query_notes_section, ""])
    return "\n".join(lines)


def _iter_markdown_links(text: str) -> list[str]:
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
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", line)
            if target.strip()
        )
    return links


def _resolve_local_link(page_path: Path, raw_target: str) -> Path | None:
    if raw_target.startswith(("http://", "https://", "mailto:")):
        return None
    if raw_target.startswith("#"):
        return None
    target_path = unquote(raw_target.split("#", 1)[0])
    if not target_path:
        return None
    return (page_path.parent / target_path).resolve()


def _page_rel_to_wiki(wiki_dir: Path, page_path: Path) -> str:
    return page_path.relative_to(wiki_dir).as_posix()


def generate_lint_report(
    persist_path: Path,
    manifest: dict[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    wiki_dir = persist_path / WIKI_DIRNAME
    generated = generated_at or datetime.now().astimezone()
    report: dict[str, Any] = {
        "version": 1,
        "generated_at": generated.isoformat(timespec="seconds"),
        "summary": {
            "stale_pages": 0,
            "orphan_pages": 0,
            "missing_links": 0,
        },
        "stale_pages": [],
        "orphan_pages": [],
        "missing_links": [],
    }

    if not wiki_dir.exists():
        return report

    markdown_pages = sorted(wiki_dir.rglob("*.md"))
    inbound_links: set[Path] = set()
    missing_links: list[dict[str, str]] = []
    page_set = {page.resolve() for page in markdown_pages}

    for page_path in markdown_pages:
        text = page_path.read_text(encoding="utf-8")
        for raw_target in _iter_markdown_links(text):
            resolved_target = _resolve_local_link(page_path, raw_target)
            if resolved_target is None:
                continue
            if resolved_target in page_set:
                inbound_links.add(resolved_target)
                continue
            if wiki_dir.resolve() in resolved_target.parents or resolved_target == wiki_dir.resolve():
                missing_links.append(
                    {
                        "page": _page_rel_to_wiki(wiki_dir, page_path),
                        "target": raw_target,
                    }
                )

    root_pages = {
        (wiki_dir / "index.md").resolve(),
        (wiki_dir / "log.md").resolve(),
    }
    orphan_pages = [
        {
            "page": _page_rel_to_wiki(wiki_dir, page_path),
        }
        for page_path in markdown_pages
        if page_path.resolve() not in root_pages and page_path.resolve() not in inbound_links
    ]

    stale_pages: list[dict[str, str]] = []
    if isinstance(manifest, dict):
        normalized_root = persist_path / str(
            manifest.get("normalized_text_dir", NORMALIZED_TEXT_DIRNAME) or NORMALIZED_TEXT_DIRNAME
        )
        for entry in _iter_file_entries(manifest):
            source = str(entry["name"])
            page_path = wiki_dir / _page_relpath_for_source(source)
            normalized_rel = str(entry.get("normalized_text", "") or "")
            normalized_path = normalized_root / normalized_rel if normalized_rel else None
            if not page_path.exists():
                stale_pages.append(
                    {
                        "page": _page_relpath_for_source(source).as_posix(),
                        "reason": "missing_page",
                        "source": source,
                    }
                )
                continue
            if normalized_path is None or not normalized_path.exists():
                continue
            page_mtime = page_path.stat().st_mtime
            normalized_mtime = normalized_path.stat().st_mtime
            if page_mtime + 1e-6 < normalized_mtime:
                stale_pages.append(
                    {
                        "page": _page_rel_to_wiki(wiki_dir, page_path),
                        "reason": "normalized_text_newer",
                        "source": source,
                    }
                )

    report["stale_pages"] = stale_pages
    report["orphan_pages"] = orphan_pages
    report["missing_links"] = missing_links
    report["summary"] = {
        "stale_pages": len(stale_pages),
        "orphan_pages": len(orphan_pages),
        "missing_links": len(missing_links),
    }

    lint_path = persist_path / LINT_REPORT_FILENAME
    lint_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _generate_wiki_unlocked(persist_path: Path, manifest: dict[str, Any]) -> dict[str, int]:
    entries = _iter_file_entries(manifest)
    build_time = str(manifest.get("build_time", "") or "unknown")
    normalized_root = persist_path / str(
        manifest.get("normalized_text_dir", NORMALIZED_TEXT_DIRNAME) or NORMALIZED_TEXT_DIRNAME
    )

    wiki_dir = persist_path / WIKI_DIRNAME
    files_dir = wiki_dir / "files"
    existing_log_text = ""
    if (wiki_dir / "log.md").exists():
        existing_log_text = (wiki_dir / "log.md").read_text(encoding="utf-8")
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
    (wiki_dir / "log.md").write_text(
        _render_log(entries, build_time, existing_log_text=existing_log_text),
        encoding="utf-8",
    )
    lint_report = generate_lint_report(persist_path=persist_path, manifest=manifest)
    return {"pages": len(entries), "lint_issues": sum(int(count) for count in lint_report["summary"].values())}


def generate_wiki(persist_path: Path, manifest: dict[str, Any]) -> dict[str, int]:
    wiki_dir = persist_path / WIKI_DIRNAME
    with _write_lock(wiki_dir):
        return _generate_wiki_unlocked(persist_path=persist_path, manifest=manifest)


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

    with _write_lock(wiki_dir):
        manifest = _load_manifest(persist_path)
        manifest = _ensure_wiki_artifacts_for_sources(
            persist_path=persist_path,
            manifest=manifest,
            sources=valid_sources,
        )
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
        generate_lint_report(persist_path=persist_path, manifest=manifest, generated_at=created)
    return {
        "note_path": str(note_path),
        "note_relpath": (Path(WIKI_DIRNAME) / note_relpath).as_posix(),
        "log_path": str(log_path),
    }
