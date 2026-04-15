from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragbot_artifacts import (
    _source_snapshot_from_file_records,
    atomic_write_text,
    current_build_config_snapshot,
)

if TYPE_CHECKING:
    from ragbot import IndexedFile


def _core():
    import ragbot as core

    return core


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
    embed_base_url: str,
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
    build_snapshot = current_build_config_snapshot(
        source_dir,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
    )
    manifest = {
        "build_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "embed_model": embed_model,
        "total_chunks": total_chunks,
        "kb_enabled": True,
        "source_dir": str(Path(source_dir).expanduser()),
        "normalized_text_dir": core.NORMALIZED_TEXT_DIRNAME,
        "symbol_index_file": core.SYMBOL_INDEX_FILENAME,
        "fulltext_index_file": core.FULLTEXT_INDEX_FILENAME,
        "document_graph_file": core.DOCUMENT_GRAPH_FILENAME,
        "entity_graph_file": core.ENTITY_GRAPH_FILENAME,
        "semantic_extract_cache_file": core.SEMANTIC_EXTRACT_CACHE_FILENAME,
        "community_index_file": core.COMMUNITY_INDEX_FILENAME,
        "graph_report_file": f"{core.REPORTS_DIRNAME}/{core.GRAPH_REPORT_FILENAME}",
        "lint_report_file": core.LINT_REPORT_FILENAME,
        "search_mode_default": os.getenv("SEARCH_MODE", core.DEFAULT_SEARCH_MODE).strip()
        or core.DEFAULT_SEARCH_MODE,
        "semantic_graph_stats": {},
        "build_snapshot": build_snapshot,
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
    source_file_records: list[dict[str, Any]] = []
    for indexed in indexed_files:
        normalized_rel = f"{indexed.rel_path}.txt"
        cache_path = normalized_dir / normalized_rel
        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            indexed.write_normalized_text(cache_path)

        stat = indexed.file_path.stat()
        source_file_records.append(
            {
                "path": indexed.rel_path,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
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
    manifest["source_snapshot"] = _source_snapshot_from_file_records(
        source_dir,
        source_file_records,
    )

    fulltext_index_path = persist_path / core.FULLTEXT_INDEX_FILENAME
    fulltext_index = core._build_fulltext_index(indexed_files)
    atomic_write_text(
        fulltext_index_path,
        json.dumps(fulltext_index, ensure_ascii=False, separators=(",", ":")),
    )

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
