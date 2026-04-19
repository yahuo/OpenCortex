from __future__ import annotations

import fnmatch
import json
import re
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from langchain_core.documents import Document

if TYPE_CHECKING:
    from ragbot import GraphExpansionResult, QueryPlan, SearchHit, SearchStepResult
    from ragbot_runtime import SearchBundle


def _core():
    import ragbot as core

    return core


def _vector_hit_from_document(doc: Document, raw_score: float | None = None) -> SearchHit:
    core = _core()
    meta = doc.metadata
    return core.SearchHit(
        source=core._normalize_source_path(meta.get("source", "")),
        match_kind="vector",
        snippet=doc.page_content,
        score=1.0 if raw_score is None else 1.0 / (1.0 + max(raw_score, 0.0)),
        line_start=meta.get("line_start"),
        line_end=meta.get("line_end"),
        metadata={
            "time_range": meta.get("time_range", ""),
            "heading": meta.get("heading", ""),
            "section_path": meta.get("section_path", []),
            "chunk_kind": meta.get("chunk_kind", ""),
        },
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
    top_k: int = 6,
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


def _merge_vector_hits(
    primary_hits: list[SearchHit],
    secondary_hits: list[SearchHit],
    limit: int,
) -> list[SearchHit]:
    return _merge_hit_lists(primary_hits, secondary_hits, limit=limit)


def _merge_hit_lists(
    primary_hits: list[SearchHit],
    secondary_hits: list[SearchHit],
    *,
    limit: int,
) -> list[SearchHit]:
    merged: list[SearchHit] = []
    seen: set[tuple[str, int | None, int | None, str]] = set()
    for hit in [*primary_hits, *secondary_hits]:
        key = hit.dedupe_key()
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)
        if len(merged) >= limit:
            break
    return merged


def _candidate_sources_from_hits(*hit_lists: list[SearchHit]) -> list[str]:
    core = _core()
    return core._dedupe_strings(
        hit.source
        for hits in hit_lists
        for hit in hits
    )


def _merge_query_plan(
    base_plan: QueryPlan,
    *,
    symbols: Iterable[str] = (),
    keywords: Iterable[str] = (),
    path_globs: Iterable[str] = (),
) -> QueryPlan:
    core = _core()
    return core.QueryPlan(
        symbols=core._dedupe_strings([*base_plan.symbols, *symbols]),
        keywords=core._dedupe_strings([*base_plan.keywords, *keywords]),
        path_globs=core._dedupe_strings([*base_plan.path_globs, *path_globs]),
        semantic_query=base_plan.semantic_query,
        reason=base_plan.reason,
    )


def _query_requests_symbol_lookup(question: str, query_plan: QueryPlan) -> bool:
    lowered = question.lower()
    if query_plan.symbols:
        return True
    symbol_phrases = (
        "哪个函数",
        "哪一个函数",
        "函数定义",
        "函数实现",
        "哪个方法",
        "哪一个方法",
        "方法定义",
        "方法实现",
        "哪个类",
        "哪一个类",
        "类定义",
        "类实现",
        "哪个接口",
        "接口定义",
        "入口函数",
        "入口方法",
        "入口类",
        "symbol",
        "which function",
        "what function",
        "function definition",
        "function implementation",
        "which method",
        "what method",
        "method definition",
        "method implementation",
        "which class",
        "what class",
        "class definition",
        "class implementation",
        "entrypoint",
    )
    return any(phrase in lowered for phrase in symbol_phrases)


def _query_requests_file_lookup(question: str, query_plan: QueryPlan) -> bool:
    lowered = question.lower()
    if query_plan.path_globs:
        return True
    file_locator_phrases = (
        "哪个文件",
        "哪一个文件",
        "在哪个文件",
        "文件名",
        "哪个文档",
        "哪一个文档",
        "哪份文档",
        "哪篇文档",
        "文档名",
        "which file",
        "what file",
        "file name",
        "which document",
        "what document",
        "document name",
        ".py",
        ".md",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
    )
    return any(phrase in lowered for phrase in file_locator_phrases)


def _query_expansion_candidates(
    bundle: SearchBundle,
    query_terms: set[str],
    question: str,
    *,
    kb: str | None,
    allowed_sources: set[str] | None,
) -> list[dict[str, Any]]:
    index = bundle.query_expansion_index or {}
    items = index.get("items")
    postings = index.get("postings")
    if not isinstance(items, list) or not isinstance(postings, dict) or not query_terms:
        return []

    lowered_question = question.lower()
    candidate_ids: set[int] = set()
    for term in query_terms:
        raw_ids = postings.get(term)
        if not isinstance(raw_ids, list):
            continue
        for item_id in raw_ids:
            if isinstance(item_id, int):
                candidate_ids.add(item_id)

    ranked: list[dict[str, Any]] = []
    for item_id in candidate_ids:
        if item_id < 0 or item_id >= len(items):
            continue
        item = items[item_id]
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "") or "")
        item_kb = str(item.get("kb", "") or "")
        if kb is not None and source and item_kb and item_kb != kb:
            continue
        if allowed_sources is not None and source and source not in allowed_sources:
            continue
        item_tokens = {
            str(token).strip().lower()
            for token in item.get("tokens", [])
            if str(token).strip()
        }
        overlap = sorted(query_terms & item_tokens)
        if not overlap:
            continue
        text = str(item.get("text", "") or "")
        exact_surface = bool(text) and text.lower() in lowered_question
        score = float(len(overlap))
        if exact_surface:
            score += 2.0
        if item_tokens and item_tokens <= query_terms:
            score += 0.75
        score += float(item.get("priority") or 0) * 0.1
        if len(overlap) == 1 and not exact_surface and len(item_tokens) > 3:
            continue
        ranked.append(
            {
                **item,
                "score": score,
                "overlap": overlap,
            }
        )

    ranked.sort(
        key=lambda item: (
            -float(item.get("score", 0.0) or 0.0),
            -int(item.get("priority", 0) or 0),
            len(str(item.get("text", "") or "")),
            str(item.get("source", "") or ""),
        )
    )
    return ranked


def _expand_query_plan_from_corpus_index(
    bundle: SearchBundle,
    question: str,
    query_plan: QueryPlan,
    *,
    kb: str | None,
    allowed_sources: set[str] | None,
) -> tuple[QueryPlan, dict[str, Any]]:
    core = _core()
    if query_plan.symbols or any(
        core._looks_like_explicit_source_term(keyword) or core._has_supported_file_suffix(keyword)
        for keyword in query_plan.keywords
    ):
        return query_plan, {"used": False, "selected": [], "added": {}}
    query_terms = set(core._fulltext_query_terms(question, query_plan=query_plan))
    if not query_terms:
        return query_plan, {"used": False, "selected": [], "added": {}}

    ranked = _query_expansion_candidates(
        bundle,
        query_terms,
        question,
        kb=kb,
        allowed_sources=allowed_sources,
    )
    if not ranked:
        return query_plan, {"used": False, "selected": [], "added": {}}

    added_symbols: list[str] = []
    added_keywords: list[str] = []
    added_path_globs: list[str] = []
    selected: list[dict[str, Any]] = []
    by_kind: dict[str, int] = {}
    wants_symbol = _query_requests_symbol_lookup(question, query_plan)
    wants_file = _query_requests_file_lookup(question, query_plan)

    for item in ranked:
        kind = str(item.get("kind", "") or "")
        if by_kind.get(kind, 0) >= 2:
            continue
        text = str(item.get("text", "") or "")
        path_glob = str(item.get("path_glob", "") or "")
        symbol = str(item.get("symbol", "") or "")
        if kind == "symbol":
            if symbol and wants_symbol:
                added_symbols.append(symbol)
        elif kind == "file":
            if text:
                added_keywords.append(text)
            if path_glob and wants_file:
                added_path_globs.append(path_glob)
        else:
            if text:
                added_keywords.append(text)
            if path_glob and wants_file and kind in {"section", "concept", "decision"}:
                added_path_globs.append(path_glob)
        if not any([text, path_glob, symbol]):
            continue
        by_kind[kind] = by_kind.get(kind, 0) + 1
        selected.append(
            {
                "kind": kind,
                "text": text,
                "source": str(item.get("source", "") or ""),
                "overlap": item.get("overlap", []),
                "score": round(float(item.get("score", 0.0) or 0.0), 3),
            }
        )
        if len(selected) >= 6:
            break

    merged_plan = _merge_query_plan(
        query_plan,
        symbols=added_symbols,
        keywords=added_keywords,
        path_globs=added_path_globs,
    )
    return merged_plan, {
        "used": merged_plan != query_plan,
        "selected": selected,
        "added": {
            "symbols": [item for item in added_symbols if item not in query_plan.symbols],
            "keywords": [item for item in added_keywords if item not in query_plan.keywords],
            "path_globs": [item for item in added_path_globs if item not in query_plan.path_globs],
        },
    }


def _preferred_symbol_for_source(bundle: SearchBundle, source: str) -> str:
    stem = Path(source).stem.lower()
    fallback = ""
    for record in bundle.symbol_index:
        if str(record.get("source", "") or "") != source:
            continue
        name = str(record.get("name", "") or "").strip()
        qualified_name = str(record.get("qualified_name") or name).strip()
        if not name:
            continue
        lowered_name = name.lower()
        lowered_qualified = qualified_name.lower()
        if lowered_name == stem or lowered_qualified == stem:
            return name
        if stem.endswith(lowered_name) or lowered_name.endswith(stem):
            fallback = fallback or name
    return fallback


def _expand_query_plan_from_vector_feedback(
    bundle: SearchBundle,
    question: str,
    query_plan: QueryPlan,
    vector_hits: list[SearchHit],
    *,
    kb: str | None,
    allowed_sources: set[str] | None,
) -> tuple[QueryPlan, dict[str, Any]]:
    core = _core()
    seed_sources = core._dedupe_strings(hit.source for hit in vector_hits[:3] if hit.source)
    if not seed_sources:
        return query_plan, {"used": False, "seed_sources": [], "added": {}}

    graph_feedback = core._expand_candidate_sources_detailed(
        bundle,
        seed_sources,
        kb=kb,
        allowed_sources=allowed_sources,
        max_hops=1,
        max_extra_sources=6,
    )
    source_hints = core._dedupe_strings([*seed_sources, *graph_feedback.expanded_sources])
    added_keywords: list[str] = []
    added_symbols: list[str] = []
    added_path_globs: list[str] = []

    wants_symbol = _query_requests_symbol_lookup(question, query_plan)
    wants_file = _query_requests_file_lookup(question, query_plan)
    for source in source_hints[:4]:
        basename = Path(source).name
        if basename and wants_file:
            added_path_globs.append(f"*{basename}*")
            added_keywords.append(basename)
        symbol_name = _preferred_symbol_for_source(bundle, source)
        if symbol_name and wants_symbol:
            added_symbols.append(symbol_name)

    if wants_symbol:
        for bridge in graph_feedback.bridge_entities:
            if str(bridge.get("type", "") or "") != "symbol":
                continue
            symbol_name = str(bridge.get("name", "") or "").strip()
            if symbol_name:
                added_symbols.append(symbol_name)

    merged_plan = _merge_query_plan(
        query_plan,
        symbols=added_symbols,
        keywords=added_keywords,
        path_globs=added_path_globs,
    )
    return merged_plan, {
        "used": merged_plan != query_plan,
        "seed_sources": seed_sources,
        "graph_strategy": graph_feedback.strategy,
        "graph_expanded_sources": graph_feedback.expanded_sources,
        "bridge_entities": graph_feedback.bridge_entities,
        "added": {
            "symbols": [item for item in added_symbols if item not in query_plan.symbols],
            "keywords": [item for item in added_keywords if item not in query_plan.keywords],
            "path_globs": [item for item in added_path_globs if item not in query_plan.path_globs],
        },
    }


def _merge_grouped_hits(*grouped_sets: dict[str, list[SearchHit]]) -> dict[str, list[SearchHit]]:
    core = _core()
    merged = {match_kind: [] for match_kind in core.RRF_WEIGHTS}
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
    core = _core()
    expanded = set(seed_sources)
    grouped: dict[Path, list[str]] = {}
    for source in seed_sources:
        source_path = Path(source)
        grouped.setdefault(source_path.parent, []).append(source_path.stem.lower())

    edge_reasons: list[dict[str, Any]] = []
    for source in core._bundle_sources(bundle, kb=kb, allowed_sources=allowed_sources):
        if source in expanded:
            continue
        candidate = Path(source)
        sibling_stems = grouped.get(candidate.parent)
        if not sibling_stems:
            continue
        stem = candidate.stem.lower()
        stem_tokens = core._stem_tokens(stem)
        for sibling in sibling_stems:
            sibling_tokens = core._stem_tokens(sibling)
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

    return core.GraphExpansionResult(
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
    core = _core()
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

    return core.GraphExpansionResult(
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
    core = _core()
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
        for neighbor in core._entity_file_neighbors(bundle, current, valid_sources=valid_sources):
            target = str(neighbor.get("to", "")).strip()
            if not target or target in seen:
                continue
            seen.add(target)
            expanded_sources.append(target)
            hop = depth + 1
            bridges = list(neighbor.get("bridges") or [])
            for bridge in bridges:
                core._add_bridge_entity(bridge_entities, seen_bridges, bridge)
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

    return core.GraphExpansionResult(
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
    core = _core()
    valid_sources = (
        set(core._bundle_sources(bundle, kb=kb, allowed_sources=allowed_sources))
        if kb is not None or allowed_sources is not None
        else None
    )
    ordered_seeds = [
        source
        for source in core._dedupe_strings(seed_sources)
        if valid_sources is None or source in valid_sources
    ]
    if not ordered_seeds:
        return core.GraphExpansionResult(set(), [], [], [], 0)

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
    core = _core()
    phrases = core._dedupe_strings(
        item
        for item in [question, query_plan.semantic_query]
        if isinstance(item, str) and len(item.strip()) >= 2
    )
    tokens = core._dedupe_strings(
        cleaned
        for item in [*query_plan.symbols, *query_plan.keywords, *query_plan.path_globs]
        for cleaned in [str(item).strip().strip("*").strip(".,:;!?()[]{}")]
        if len(cleaned) >= 2 and cleaned.lower() not in core.GRAPH_STOPWORDS
    )
    return phrases, tokens


def _wiki_kind_priority(kind: str) -> int:
    return {"query": 4, "entity": 3, "community": 2, "file": 1, "index": 0}.get(kind, 0)


def _wiki_source_ref_matches_query(source: str, query_plan: QueryPlan) -> bool:
    core = _core()
    source_lower = source.lower()
    basename_lower = Path(source).name.lower()
    specific_terms = core._dedupe_strings(
        term.lower()
        for term in [*query_plan.keywords, *query_plan.symbols]
        if isinstance(term, str) and core._looks_like_explicit_source_term(term)
    )
    specific_patterns = core._dedupe_strings(
        pattern.lower()
        for pattern in query_plan.path_globs
        if isinstance(pattern, str)
        and pattern.strip()
        and not core._is_extension_only_glob(pattern)
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
    core = _core()
    if not bundle.wiki_pages:
        return []

    valid_sources = (
        set(core._bundle_sources(bundle, kb=kb, allowed_sources=allowed_sources))
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
            source_refs = [
                source for source in source_refs if _wiki_source_ref_matches_query(source, query_plan)
            ]
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

        deduped_terms = core._dedupe_strings(matched_terms)
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


def _has_high_confidence_lexical_hits(*hit_lists: list[SearchHit]) -> bool:
    return any(
        hit.metadata.get("exact_path") or hit.metadata.get("exact_symbol")
        for hits in hit_lists
        for hit in hits
    )


def _vector_scope_is_narrowed(
    bundle: SearchBundle,
    *,
    kb: str | None,
    allowed_sources: set[str] | None,
    candidate_scope: set[str],
) -> bool:
    core = _core()
    if not candidate_scope:
        return False
    full_scope = set(core._bundle_sources(bundle, kb=kb, allowed_sources=allowed_sources))
    return bool(full_scope) and candidate_scope < full_scope


def _should_broaden_vector_recall(
    bundle: SearchBundle,
    *,
    kb: str | None,
    allowed_sources: set[str] | None,
    candidate_scope: set[str],
    fused_hits: list[SearchHit],
    glob_hits: list[SearchHit],
    grep_hits: list[SearchHit],
    ast_hits: list[SearchHit],
) -> bool:
    if not _vector_scope_is_narrowed(
        bundle,
        kb=kb,
        allowed_sources=allowed_sources,
        candidate_scope=candidate_scope,
    ):
        return False
    if _has_high_confidence_lexical_hits(glob_hits, grep_hits, ast_hits):
        return False
    if not fused_hits:
        return True
    top_sources = {hit.source for hit in fused_hits[:3]}
    return len(top_sources) < min(2, len(fused_hits))


def _should_apply_vector_feedback(
    question: str,
    query_plan: QueryPlan,
    grouped_hits: dict[str, list[SearchHit]],
    fused_hits: list[SearchHit],
) -> bool:
    if not fused_hits or not grouped_hits.get("vector"):
        return False
    if query_plan.symbols or query_plan.path_globs:
        return False
    if _has_high_confidence_lexical_hits(
        grouped_hits.get("glob", []),
        grouped_hits.get("grep", []),
        grouped_hits.get("ast", []),
    ):
        return False
    if grouped_hits.get("grep") or grouped_hits.get("ast"):
        return False
    if not _query_requests_symbol_lookup(question, query_plan) and fused_hits[0].match_kind != "vector":
        return False
    return True


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
    core = _core()
    effective_plan, corpus_expansion = core._expand_query_plan_from_corpus_index(
        bundle,
        question,
        query_plan,
        kb=kb,
        allowed_sources=allowed_sources,
    )
    wiki_hits = core._wiki_search(
        bundle,
        question=question,
        query_plan=effective_plan,
        kb=kb,
        allowed_sources=allowed_sources,
    )
    wiki_seed_sources = core._dedupe_strings(
        source
        for hit in wiki_hits
        for source in hit.get("source_refs", [])
        if isinstance(source, str) and source
    )
    wiki_scope = set(wiki_seed_sources)
    glob_hits = core.glob_search(bundle, effective_plan.path_globs, kb=kb, allowed_sources=allowed_sources)
    glob_scope = wiki_scope | set(core._candidate_sources_from_hits(glob_hits))
    grep_scope = glob_scope or allowed_sources
    grep_hits = core.grep_search(
        bundle,
        keywords=[*effective_plan.keywords, *effective_plan.symbols],
        kb=kb,
        allowed_sources=grep_scope,
    )
    grep_fallback_used = False
    should_retry_grep = not grep_hits and bool(glob_scope)
    if should_retry_grep:
        grep_hits = core.grep_search(
            bundle,
            keywords=[*effective_plan.keywords, *effective_plan.symbols],
            kb=kb,
            allowed_sources=allowed_sources,
        )
        grep_fallback_used = True
    ast_hits = core.ast_search(bundle, effective_plan, kb=kb, allowed_sources=allowed_sources)
    bm25_hits = core.bm25_search(
        bundle,
        question=effective_plan.semantic_query or question,
        query_plan=effective_plan,
        kb=kb,
        allowed_sources=allowed_sources,
        top_k=max(top_k * 2, 8),
    )
    graph_seed_sources = core._dedupe_strings(
        [
            *wiki_seed_sources,
            *core._candidate_sources_from_hits(glob_hits[:3], grep_hits[:3], ast_hits[:3], bm25_hits[:3]),
        ]
    )
    graph_expansion = core._expand_candidate_sources_detailed(
        bundle,
        graph_seed_sources,
        kb=kb,
        allowed_sources=allowed_sources,
        max_hops=graph_max_hops,
        max_extra_sources=graph_max_extra_sources,
    )
    narrowed_vector_scope = wiki_scope | set(
        core._candidate_sources_from_hits(glob_hits, grep_hits[:4], ast_hits[:4], bm25_hits[:4])
    )
    if graph_expansion.sources:
        narrowed_vector_scope = narrowed_vector_scope | graph_expansion.sources
    if not narrowed_vector_scope:
        narrowed_vector_scope = allowed_sources or set()

    scoped_vector_hits = core.vector_search(
        bundle,
        query=effective_plan.semantic_query or question,
        kb=kb,
        top_k=max(top_k * 2, 8),
        candidate_sources=narrowed_vector_scope or None,
    )

    grouped = {
        "glob": glob_hits,
        "grep": grep_hits,
        "ast": ast_hits,
        "bm25": bm25_hits,
        "vector": scoped_vector_hits,
    }
    fused = core._finalize_hits(grouped, top_k=top_k)
    vector_fallback_used = False
    vector_fallback_reason = ""
    if _should_broaden_vector_recall(
        bundle,
        kb=kb,
        allowed_sources=allowed_sources,
        candidate_scope=narrowed_vector_scope,
        fused_hits=fused,
        glob_hits=glob_hits,
        grep_hits=grep_hits,
        ast_hits=ast_hits,
    ):
        broader_vector_hits = core.vector_search(
            bundle,
            query=effective_plan.semantic_query or question,
            kb=kb,
            top_k=max(top_k * 2, 8),
            candidate_sources=allowed_sources or None,
        )
        merged_vector_hits = _merge_vector_hits(
            broader_vector_hits,
            scoped_vector_hits,
            limit=max(top_k * 3, 12),
        )
        if len(merged_vector_hits) > len(scoped_vector_hits):
            grouped["vector"] = merged_vector_hits
            fused = core._finalize_hits(grouped, top_k=top_k)
            vector_fallback_used = True
            vector_fallback_reason = "narrow_scope_low_confidence"

    feedback_plan = effective_plan
    feedback_expansion: dict[str, Any] = {"used": False, "seed_sources": [], "added": {}}
    feedback_candidate_sources: list[str] = []
    if _should_apply_vector_feedback(question, effective_plan, grouped, fused):
        feedback_plan, feedback_expansion = core._expand_query_plan_from_vector_feedback(
            bundle,
            question,
            effective_plan,
            grouped["vector"],
            kb=kb,
            allowed_sources=allowed_sources,
        )
    if feedback_plan != effective_plan:
        feedback_glob_hits = (
            core.glob_search(bundle, feedback_plan.path_globs, kb=kb, allowed_sources=allowed_sources)
            if feedback_plan.path_globs != effective_plan.path_globs
            else []
        )
        feedback_grep_hits = core.grep_search(
            bundle,
            keywords=[*feedback_plan.keywords, *feedback_plan.symbols],
            kb=kb,
            allowed_sources=allowed_sources,
        )
        feedback_ast_hits = core.ast_search(bundle, feedback_plan, kb=kb, allowed_sources=allowed_sources)
        feedback_bm25_hits = core.bm25_search(
            bundle,
            question=feedback_plan.semantic_query or question,
            query_plan=feedback_plan,
            kb=kb,
            allowed_sources=allowed_sources,
            top_k=max(top_k * 2, 8),
        )
        grouped["glob"] = _merge_hit_lists(feedback_glob_hits, grouped["glob"], limit=max(top_k * 2, 8))
        grouped["grep"] = _merge_hit_lists(feedback_grep_hits, grouped["grep"], limit=max(top_k * 2, 8))
        grouped["ast"] = _merge_hit_lists(feedback_ast_hits, grouped["ast"], limit=max(top_k * 2, 8))
        grouped["bm25"] = _merge_hit_lists(feedback_bm25_hits, grouped["bm25"], limit=max(top_k * 2, 8))
        feedback_candidate_sources = core._candidate_sources_from_hits(
            feedback_glob_hits[:4],
            feedback_grep_hits[:4],
            feedback_ast_hits[:4],
            feedback_bm25_hits[:4],
        )
        fused = core._finalize_hits(grouped, top_k=top_k)

    trace = {
        "step": step_name,
        "query_plan": {
            "symbols": effective_plan.symbols,
            "keywords": effective_plan.keywords,
            "path_globs": effective_plan.path_globs,
            "semantic_query": effective_plan.semantic_query,
            "reason": effective_plan.reason,
        },
        "query_expansion": corpus_expansion,
        "feedback_query_expansion": feedback_expansion,
        "feedback_query_plan": {
            "symbols": feedback_plan.symbols,
            "keywords": feedback_plan.keywords,
            "path_globs": feedback_plan.path_globs,
            "semantic_query": feedback_plan.semantic_query,
            "reason": feedback_plan.reason,
        },
        "retrievers": {name: len(hits) for name, hits in grouped.items()},
        "candidate_scope": sorted(narrowed_vector_scope) if narrowed_vector_scope else [],
        "feedback_candidate_sources": feedback_candidate_sources,
        "wiki_hits": wiki_hits,
        "wiki_scope": wiki_seed_sources,
        "grep_fallback_used": grep_fallback_used,
        "vector_scope_narrowed": _vector_scope_is_narrowed(
            bundle,
            kb=kb,
            allowed_sources=allowed_sources,
            candidate_scope=narrowed_vector_scope,
        ),
        "vector_fallback_used": vector_fallback_used,
        "vector_fallback_reason": vector_fallback_reason,
        "graph_strategy": graph_expansion.strategy,
        "graph_seed_sources": graph_expansion.seed_sources,
        "graph_expanded_sources": graph_expansion.expanded_sources,
        "graph_edge_reasons": graph_expansion.edge_reasons,
        "graph_bridge_entities": graph_expansion.bridge_entities,
        "graph_hops": graph_expansion.hops,
        "top_sources": [hit.source for hit in fused[:3]],
    }
    return core.SearchStepResult(grouped_hits=grouped, hits=fused, trace=trace)


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
    core = _core()
    evidence_lines = []
    for hit in hits[:3]:
        location = core._line_range_label(hit.line_start, hit.line_end)
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
    core = _core()
    if not llm_api_key.strip():
        return None
    try:
        llm = core.make_llm(
            api_key=llm_api_key,
            model=llm_model,
            base_url=llm_base_url,
            temperature=0.0,
        )
        response = llm.invoke(_planner_prompt(question, hits))
        content = core._chunk_to_text(response)
        payload = _extract_json_blob(content)
        symbols = payload.get("symbols") or []
        keywords = payload.get("keywords") or []
        path_globs = payload.get("path_globs") or []
        semantic_query = payload.get("semantic_query") or question
        if not any([symbols, keywords, path_globs, semantic_query]):
            return None
        return core.QueryPlan(
            symbols=core._dedupe_strings(str(item) for item in symbols),
            keywords=core._dedupe_strings(str(item) for item in keywords),
            path_globs=core._dedupe_strings(str(item) for item in path_globs),
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
    core = _core()
    if followup is None:
        return base
    return core.QueryPlan(
        symbols=core._dedupe_strings([*base.symbols, *followup.symbols]),
        keywords=core._dedupe_strings([*base.keywords, *followup.keywords]),
        path_globs=core._dedupe_strings([*base.path_globs, *followup.path_globs]),
        semantic_query=followup.semantic_query.strip() or question,
        reason=followup.reason or base.reason,
    )


def _sources_from_hits(results: list[SearchHit]) -> list[dict[str, Any]]:
    core = _core()
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    for hit in results:
        key = (hit.source, hit.line_start, hit.line_end)
        if key in seen:
            continue
        seen.add(key)
        time_range = hit.metadata.get("time_range") or core._line_range_label(hit.line_start, hit.line_end)
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
    core = _core()
    bridge_entities: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for step in search_trace:
        raw_entities = step.get("graph_bridge_entities", [])
        if not isinstance(raw_entities, list):
            continue
        for raw_entity in raw_entities:
            if not isinstance(raw_entity, dict):
                continue
            core._add_bridge_entity(bridge_entities, seen, raw_entity)
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
    core = _core()
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
            search_bundle.manifest.get("graph_report_file", f"{core.REPORTS_DIRNAME}/{core.GRAPH_REPORT_FILENAME}")
            or f"{core.REPORTS_DIRNAME}/{core.GRAPH_REPORT_FILENAME}"
        ),
        "fulltext_index_path": str(
            search_bundle.manifest.get("fulltext_index_file", core.FULLTEXT_INDEX_FILENAME)
            or core.FULLTEXT_INDEX_FILENAME
        ),
        "community_index_path": str(
            search_bundle.manifest.get("community_index_file", core.COMMUNITY_INDEX_FILENAME)
            or core.COMMUNITY_INDEX_FILENAME
        ),
        "lint_report_path": str(
            search_bundle.manifest.get("lint_report_file", core.LINT_REPORT_FILENAME) or core.LINT_REPORT_FILENAME
        ),
    }


def _build_context_and_sources(results: list[SearchHit]) -> tuple[str, list[dict[str, Any]]]:
    core = _core()
    context_parts = []
    for index, hit in enumerate(results, start=1):
        location = core._line_range_label(hit.line_start, hit.line_end)
        title = hit.source if not location else f"{hit.source}:{location}"
        context_parts.append(f"[证据{index}] {title}\n{hit.snippet}")
    context = "\n\n".join(context_parts) if context_parts else "未找到直接匹配的参考资料。"
    return context, _sources_from_hits(results)


def retrieve(
    question: str,
    search_bundle: SearchBundle,
    kb: str | None = None,
    mode: str | None = None,
    top_k: int = 6,
    llm_api_key: str = "",
    llm_model: str = "",
    llm_base_url: str = "",
) -> dict[str, Any]:
    core = _core()
    search_mode = core._search_mode(mode)
    base_plan = core._extract_query_plan(question)
    trace: list[dict[str, Any]] = []

    pool_size = max(top_k, core._rerank_top_n()) if core._rerank_enabled() else top_k

    def apply_rerank(hits: list[Any]) -> tuple[list[Any], dict[str, Any]]:
        return core.llm_rerank(
            question,
            hits,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            keep=top_k,
        )

    if search_mode == "vector":
        vector_hits = core.vector_search(search_bundle, question, kb=kb, top_k=pool_size)
        vector_hits, rerank_trace = apply_rerank(vector_hits)
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
                "rerank": rerank_trace,
                "stopped": True,
                "stop_reason": "vector_mode",
            }
        )
        return {
            "hits": vector_hits,
            "context": context,
            "sources": sources,
            "search_trace": trace,
            "wiki_trace": core._collect_wiki_trace(trace),
            "bridge_entities": [],
        }

    step1_result = core._run_search_step(
        question=question,
        bundle=search_bundle,
        query_plan=base_plan,
        kb=kb,
        top_k=pool_size,
        allowed_sources=None,
        step_name="step1",
    )
    stopped, reason = core._should_stop_after_first_step(step1_result.hits)
    step1_result.trace["stopped"] = stopped or search_mode != "agentic" or core._search_max_steps() <= 1
    step1_result.trace["stop_reason"] = reason if step1_result.trace["stopped"] else "followup_requested"
    trace.append(step1_result.trace)

    final_hits = step1_result.hits
    if search_mode == "agentic" and not stopped and core._search_max_steps() > 1:
        prefilter_expansion = core._expand_candidate_sources_detailed(
            search_bundle,
            core._dedupe_strings(hit.source for hit in step1_result.hits),
            kb=kb,
            max_hops=2,
            max_extra_sources=20,
        )
        planner_plan = core._call_retrieval_planner(
            question=question,
            hits=step1_result.hits,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
        )
        merged_plan = core._merge_plans(base_plan, planner_plan, question)
        step2_result = core._run_search_step(
            question=question,
            bundle=search_bundle,
            query_plan=merged_plan,
            kb=kb,
            top_k=pool_size,
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
        final_hits = core._finalize_hits(
            core._merge_grouped_hits(step1_result.grouped_hits, step2_result.grouped_hits),
            top_k=pool_size,
        )

    final_hits, rerank_trace = apply_rerank(final_hits)
    if trace:
        trace[-1]["rerank"] = rerank_trace

    context, sources = _build_context_and_sources(final_hits)
    bridge_entities = core._collect_bridge_entities(trace)
    return {
        "hits": final_hits,
        "context": context,
        "sources": sources,
        "search_trace": trace,
        "wiki_trace": core._collect_wiki_trace(trace),
        "bridge_entities": bridge_entities,
    }


def ask_stream(
    question: str,
    search_bundle: SearchBundle,
    llm_api_key: str,
    llm_model: str,
    llm_base_url: str,
    top_k: int = 6,
    kb: str | None = None,
    search_mode: str | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    core = _core()
    retrieval = core.retrieve(
        question=question,
        search_bundle=search_bundle,
        kb=kb,
        mode=search_mode,
        top_k=top_k,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
    )

    llm = core.make_llm(api_key=llm_api_key, model=llm_model, base_url=llm_base_url)
    prompt = core._PROMPT_TEMPLATE.format(context=retrieval["context"], question=question)

    def answer_stream():
        for chunk in llm.stream(prompt):
            text = core._chunk_to_text(chunk)
            if text:
                yield text

    result: dict[str, Any] = {
        "answer_stream": answer_stream(),
        "sources": retrieval["sources"],
    }
    if debug:
        wiki_trace = retrieval.get("wiki_trace")
        if not isinstance(wiki_trace, list):
            wiki_trace = core._collect_wiki_trace(retrieval.get("search_trace", []))
        result["search_trace"] = retrieval["search_trace"]
        result["wiki_trace"] = wiki_trace
        result["bridge_entities"] = retrieval.get("bridge_entities", [])
        result["artifacts"] = core._bundle_artifacts_summary(search_bundle)
    return result
