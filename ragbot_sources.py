from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from langchain_core.documents import Document

if TYPE_CHECKING:
    from ragbot import IndexedFile


def _core():
    import ragbot as core

    return core


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
            if core._should_ignore_relative_path(rel_child, extra_patterns):
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in filenames:
            path = current_dir / filename
            rel_path = path.relative_to(source_dir)
            if core._should_ignore_relative_path(rel_path, extra_patterns):
                continue
            if path.suffix.lower() in core.SUPPORTED_TEXT_SUFFIXES:
                files.append(path)

    return sorted(files)


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
                        "heading": chunk.section_path[-1] if chunk.section_path else "",
                        "section_path": list(chunk.section_path),
                        "chunk_kind": chunk.kind,
                    },
                )
            )

    return documents, indexed_files


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
                    "heading": chunk.section_path[-1] if chunk.section_path else "",
                    "section_path": list(chunk.section_path),
                    "chunk_kind": chunk.kind,
                },
            )
