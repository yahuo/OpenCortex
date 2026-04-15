from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ragbot import IndexedFile


def _core():
    import ragbot as core

    return core


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
