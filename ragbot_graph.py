from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any


def _core():
    import ragbot as core

    return core


def _project_file_relationships(
    file_sources: list[str],
    document_graph: dict[str, Any],
    entity_graph: dict[str, Any],
) -> list[dict[str, Any]]:
    core = _core()
    file_sources = sorted(
        {
            core._normalize_source_path(source)
            for source in file_sources
            if core._normalize_source_path(source)
        }
    )
    valid_sources = set(file_sources)
    relationships: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def add(
        source: str,
        target: str,
        kind: str,
        reason: str,
        origin: str,
        bridges: list[dict[str, Any]] | None = None,
    ) -> None:
        source = core._normalize_source_path(source)
        target = core._normalize_source_path(target)
        if not source or not target or source == target:
            return
        if source not in valid_sources or target not in valid_sources:
            return
        bridge_entities = bridges or []
        key = (
            source,
            target,
            kind,
            reason,
            origin,
            tuple((item.get("id", ""), item.get("relation", "")) for item in bridge_entities),
        )
        if key in seen:
            return
        seen.add(key)
        relationships.append(
            {
                "source": source,
                "target": target,
                "kind": kind,
                "reason": reason,
                "origin": origin,
                "bridges": bridge_entities,
            }
        )

    raw_neighbors = document_graph.get("neighbors", {})
    if isinstance(raw_neighbors, dict):
        for source, edges in raw_neighbors.items():
            normalized_source = core._normalize_source_path(source)
            if not normalized_source or not isinstance(edges, list):
                continue
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                add(
                    source=normalized_source,
                    target=str(edge.get("target", "") or ""),
                    kind=str(edge.get("kind", "") or ""),
                    reason=str(edge.get("reason", "") or ""),
                    origin="document_graph",
                )

    entity_nodes_by_id: dict[str, dict[str, Any]] = {}
    raw_nodes = entity_graph.get("nodes", [])
    if isinstance(raw_nodes, list):
        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict):
                continue
            normalized_node = core._normalize_entity_node_payload(raw_node)
            if normalized_node is None:
                continue
            entity_nodes_by_id[str(normalized_node["id"])] = normalized_node

    normalized_entity_edges: list[dict[str, Any]] = []
    raw_edges = entity_graph.get("edges", [])
    if isinstance(raw_edges, list):
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                continue
            normalized_edge = core._normalize_entity_edge_payload(raw_edge)
            if normalized_edge is None:
                continue
            normalized_entity_edges.append(normalized_edge)
    semantic_attachment_files = core._entity_attachment_files(
        entity_nodes_by_id,
        normalized_entity_edges,
    )
    attachment_node_ids = set(semantic_attachment_files.keys())

    for normalized_edge in normalized_entity_edges:
        source_id = str(normalized_edge["source"])
        target_id = str(normalized_edge["target"])
        source_node = entity_nodes_by_id.get(source_id)
        target_node = entity_nodes_by_id.get(target_id)
        if not source_node or not target_node:
            continue
        source_files = set()
        target_files = set()
        source_file = core._entity_node_file_source(source_node)
        target_file = core._entity_node_file_source(target_node)
        evidence_source = core._entity_edge_evidence_source(normalized_edge)
        if source_file:
            source_files.add(source_file)
        elif source_id in attachment_node_ids:
            source_files |= semantic_attachment_files.get(source_id, set())
            if (
                str(source_node.get("type", "") or "") == "query_note"
                and evidence_source
                and evidence_source in source_files
            ):
                source_files = {evidence_source}
        if target_file:
            target_files.add(target_file)
        elif target_id in attachment_node_ids:
            target_files |= semantic_attachment_files.get(target_id, set())
            if (
                str(target_node.get("type", "") or "") == "query_note"
                and evidence_source
                and evidence_source in target_files
            ):
                target_files = {evidence_source}
        if not source_files or not target_files:
            continue
        bridges: list[dict[str, Any]] = []
        if source_node.get("type") != "file":
            bridges.append(
                core._entity_bridge_payload(
                    source_node,
                    relation=str(normalized_edge.get("type", "") or ""),
                )
            )
        if target_node.get("type") != "file":
            bridges.append(
                core._entity_bridge_payload(
                    target_node,
                    relation=str(normalized_edge.get("type", "") or ""),
                )
            )
        for projected_source in sorted(source_files):
            for projected_target in sorted(target_files):
                if projected_source == projected_target:
                    continue
                add(
                    source=projected_source,
                    target=projected_target,
                    kind=str(normalized_edge.get("type", "") or ""),
                    reason=str(normalized_edge.get("reason", "") or ""),
                    origin="entity_graph",
                    bridges=bridges,
                )

    return relationships


def _build_community_index(
    file_sources: list[str],
    document_graph: dict[str, Any],
    entity_graph: dict[str, Any],
    semantic_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    core = _core()
    file_sources = sorted(
        {
            core._normalize_source_path(source)
            for source in file_sources
            if core._normalize_source_path(source)
        }
    )
    relationships = _project_file_relationships(file_sources, document_graph, entity_graph)

    strong_adjacency: dict[str, set[str]] = {source: set() for source in file_sources}
    all_adjacency: dict[str, set[str]] = {source: set() for source in file_sources}
    for relation in relationships:
        source = relation["source"]
        target = relation["target"]
        all_adjacency.setdefault(source, set()).add(target)
        all_adjacency.setdefault(target, set()).add(source)
        if relation["kind"] in core.COMMUNITY_STRONG_EDGE_TYPES:
            strong_adjacency.setdefault(source, set()).add(target)
            strong_adjacency.setdefault(target, set()).add(source)

    communities_raw: list[list[str]] = []
    seen_sources: set[str] = set()
    for source in file_sources:
        if source in seen_sources:
            continue
        queue = deque([source])
        component: list[str] = []
        seen_sources.add(source)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(strong_adjacency.get(current, set())):
                if neighbor in seen_sources:
                    continue
                seen_sources.add(neighbor)
                queue.append(neighbor)
        communities_raw.append(sorted(component))

    communities_raw.sort(key=lambda items: (-len(items), items[0] if items else ""))
    file_to_community: dict[str, str] = {}
    community_ids: list[str] = []
    for index, files in enumerate(communities_raw, start=1):
        community_id = f"community-{index:03d}"
        community_ids.append(community_id)
        for source in files:
            file_to_community[source] = community_id

    entity_nodes = {
        str(node.get("id", "")): node
        for node in entity_graph.get("nodes", [])
        if isinstance(node, dict) and node.get("id")
    }
    entity_edges = [
        edge
        for edge in entity_graph.get("edges", [])
        if isinstance(edge, dict)
    ]
    semantic_edge_counts: dict[str, int] = {}
    normalized_entity_edges = [
        normalized_edge
        for edge in entity_edges
        if (normalized_edge := core._normalize_entity_edge_payload(edge)) is not None
    ]
    semantic_attachment_files = core._entity_attachment_files(
        entity_nodes,
        normalized_entity_edges,
    )
    attachment_node_ids = set(semantic_attachment_files.keys())
    for edge in normalized_entity_edges:
        source_id = str(edge.get("source", "") or "")
        target_id = str(edge.get("target", "") or "")
        if source_id in attachment_node_ids:
            semantic_edge_counts[source_id] = semantic_edge_counts.get(source_id, 0) + 1
        if target_id in attachment_node_ids:
            semantic_edge_counts[target_id] = semantic_edge_counts.get(target_id, 0) + 1

    communities: list[dict[str, Any]] = []
    for community_id, files in zip(community_ids, communities_raw):
        file_set = set(files)
        kbs = sorted({core._extract_kb(source) for source in files if core._extract_kb(source)})
        file_degrees = {
            source: len({neighbor for neighbor in all_adjacency.get(source, set()) if neighbor in file_set})
            for source in files
        }
        top_files = [
            {
                "source": source,
                "degree": file_degrees[source],
                "kb": core._extract_kb(source),
            }
            for source in sorted(files, key=lambda item: (-file_degrees[item], item))
        ][: core.COMMUNITY_TOP_FILES]

        symbol_scores: dict[str, tuple[dict[str, Any], int]] = {}
        for node_id, node in entity_nodes.items():
            if node.get("type") != "symbol":
                continue
            if node.get("symbol_kind") in {"module", "import"}:
                continue
            source = core._entity_node_file_source(node)
            if source not in file_set:
                continue
            score = 0
            for edge in entity_edges:
                if edge.get("source") == node_id or edge.get("target") == node_id:
                    score += 1
            symbol_scores[node_id] = (node, score)

        top_symbols = [
            {
                "id": str(node.get("id", "")),
                "name": str(node.get("qualified_name") or node.get("name") or ""),
                "source": core._entity_node_file_source(node),
                "score": score,
            }
            for node, score in sorted(
                symbol_scores.values(),
                key=lambda item: (-item[1], str(item[0].get("qualified_name") or item[0].get("name") or "")),
            )
        ][: core.COMMUNITY_TOP_SYMBOLS]
        top_concepts = [
            {
                "id": node_id,
                "name": str(node.get("name") or ""),
                "file_count": len(attached_files & file_set),
                "score": len(attached_files & file_set) * 100 + semantic_edge_counts.get(node_id, 0),
            }
            for node_id, node in entity_nodes.items()
            if node.get("type") == "concept"
            and (attached_files := semantic_attachment_files.get(node_id, set()))
            and attached_files & file_set
        ]
        top_concepts = sorted(
            top_concepts,
            key=lambda item: (-int(item["file_count"]), -int(item["score"]), str(item["name"])),
        )[: core.COMMUNITY_TOP_SYMBOLS]
        top_decisions = [
            {
                "id": node_id,
                "name": str(node.get("name") or ""),
                "file_count": len(attached_files & file_set),
                "score": len(attached_files & file_set) * 100 + semantic_edge_counts.get(node_id, 0),
            }
            for node_id, node in entity_nodes.items()
            if node.get("type") == "decision"
            and (attached_files := semantic_attachment_files.get(node_id, set()))
            and attached_files & file_set
        ]
        top_decisions = sorted(
            top_decisions,
            key=lambda item: (-int(item["file_count"]), -int(item["score"]), str(item["name"])),
        )[: core.COMMUNITY_TOP_SYMBOLS]
        top_query_notes = [
            {
                "id": node_id,
                "name": str(node.get("name") or ""),
                "file_count": len(attached_files & file_set),
                "score": len(attached_files & file_set) * 100 + semantic_edge_counts.get(node_id, 0),
                "note_relpath": str(node.get("note_relpath") or ""),
            }
            for node_id, node in entity_nodes.items()
            if node.get("type") == "query_note"
            and (attached_files := semantic_attachment_files.get(node_id, set()))
            and attached_files & file_set
        ]
        top_query_notes = sorted(
            top_query_notes,
            key=lambda item: (-int(item["file_count"]), -int(item["score"]), str(item["name"])),
        )[: core.COMMUNITY_TOP_QUERY_NOTES]

        label = (
            top_symbols[0]["name"]
            if top_symbols
            else top_concepts[0]["name"]
            if top_concepts
            else top_query_notes[0]["name"]
            if top_query_notes
            else Path(top_files[0]["source"]).stem if top_files else community_id
        )
        suggested_queries: list[str] = []
        if top_symbols:
            suggested_queries.append(f"{top_symbols[0]['name']} 在哪里实现？")
        if top_concepts:
            suggested_queries.append(f"{top_concepts[0]['name']} 在哪些文件里被讨论？")
        if top_decisions:
            suggested_queries.append(f"为什么采用 {top_decisions[0]['name']}？")
        if top_query_notes:
            suggested_queries.append(f"已保存问题“{top_query_notes[0]['name']}”涉及哪些文件？")
        if top_files:
            suggested_queries.append(f"{Path(top_files[0]['source']).stem} 相关流程是什么？")
        if kbs:
            suggested_queries.append(f"{kbs[0]} 这个社区主要关注什么？")

        communities.append(
            {
                "id": community_id,
                "label": label,
                "size": len(files),
                "files": files,
                "kbs": kbs,
                "top_files": top_files,
                "top_symbols": top_symbols,
                "top_concepts": top_concepts,
                "top_decisions": top_decisions,
                "top_query_notes": top_query_notes,
                "suggested_queries": suggested_queries[:3],
            }
        )

    file_degree_overall = {
        source: len(neighbors)
        for source, neighbors in all_adjacency.items()
    }
    god_nodes: list[dict[str, Any]] = [
        {
            "id": core._entity_file_node_id(source),
            "type": "file",
            "name": Path(source).name,
            "source": source,
            "degree": file_degree_overall.get(source, 0),
        }
        for source in sorted(file_sources, key=lambda item: (-file_degree_overall.get(item, 0), item))
    ][: core.COMMUNITY_TOP_GOD_NODES]

    bridges: list[dict[str, Any]] = []
    for relation in relationships:
        source = relation["source"]
        target = relation["target"]
        source_community = file_to_community.get(source)
        target_community = file_to_community.get(target)
        if not source_community or not target_community or source_community == target_community:
            continue
        bridges.append(
            {
                "source_community": source_community,
                "target_community": target_community,
                "source": source,
                "target": target,
                "kind": relation["kind"],
                "reason": relation["reason"],
                "origin": relation["origin"],
                "bridges": relation["bridges"],
            }
        )

    bridges.sort(
        key=lambda item: (
            0 if item["origin"] == "entity_graph" else 1,
            core._entity_expansion_priority(item["kind"]),
            item["source_community"],
            item["target_community"],
            item["source"],
            item["target"],
        )
    )
    concept_count = sum(1 for node in entity_nodes.values() if node.get("type") == "concept")
    decision_count = sum(1 for node in entity_nodes.values() if node.get("type") == "decision")
    query_note_count = sum(1 for node in entity_nodes.values() if node.get("type") == "query_note")
    semantic_summary = {
        "enabled": bool((semantic_stats or {}).get("enabled")),
        "disabled_reason": str((semantic_stats or {}).get("disabled_reason", "") or ""),
        "concept_count": concept_count,
        "decision_count": decision_count,
        "query_note_count": query_note_count,
        "semantic_node_count": concept_count + decision_count,
        "semantic_edge_count": sum(
            1
            for edge in entity_edges
            if str(edge.get("type", "") or "") in {"semantically_related", "rationale_for"}
        ),
        "cached_sections": int((semantic_stats or {}).get("cached_sections") or 0),
        "extracted_sections": int((semantic_stats or {}).get("extracted_sections") or 0),
        "failed_sections": int((semantic_stats or {}).get("failed_sections") or 0),
        "api_calls": int((semantic_stats or {}).get("api_calls") or 0),
        "total_tokens": int((semantic_stats or {}).get("total_tokens") or 0),
        "duration_seconds": float((semantic_stats or {}).get("duration_seconds") or 0.0),
    }

    return {
        "version": 1,
        "file_count": len(file_sources),
        "relationship_count": len(relationships),
        "community_count": len(communities),
        "semantic_summary": semantic_summary,
        "communities": communities,
        "file_to_community": file_to_community,
        "god_nodes": god_nodes,
        "bridges": bridges[: core.COMMUNITY_TOP_BRIDGES],
    }


def _format_report_list(items: list[str], empty_text: str) -> list[str]:
    if not items:
        return [f"- {empty_text}"]
    return [f"- {item}" for item in items]


def _render_graph_report(
    community_index: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    build_time = str(manifest.get("build_time", "") or "unknown")
    source_dir = str(manifest.get("source_dir", "") or "unknown")
    file_count = int(community_index.get("file_count") or 0)
    relationship_count = int(community_index.get("relationship_count") or 0)
    community_count = int(community_index.get("community_count") or 0)
    communities = community_index.get("communities", [])
    god_nodes = community_index.get("god_nodes", [])
    bridges = community_index.get("bridges", [])
    semantic_summary = (
        community_index.get("semantic_summary")
        if isinstance(community_index.get("semantic_summary"), dict)
        else {}
    )

    semantic_lines = [
        "# 图谱报告",
        "",
        "此报告由构建阶段自动生成，当前用于帮助理解知识库结构。",
        "",
        f"- 最近构建：{build_time}",
        f"- 源目录：{source_dir}",
        f"- 文件总数：{file_count}",
        f"- 社区数量：{community_count}",
        f"- 关系总数：{relationship_count}",
        "",
        "## 语义抽取",
        "",
        f"- 启用状态：{'enabled' if semantic_summary.get('enabled') else 'disabled'}",
    ]
    if semantic_summary.get("disabled_reason"):
        semantic_lines.append(f"- 禁用原因：{semantic_summary.get('disabled_reason')}")
    semantic_lines.extend(
        [
            f"- 语义节点：{int(semantic_summary.get('semantic_node_count') or 0)}",
            f"- Concepts：{int(semantic_summary.get('concept_count') or 0)}",
            f"- Decisions：{int(semantic_summary.get('decision_count') or 0)}",
            f"- Query Notes：{int(semantic_summary.get('query_note_count') or 0)}",
            f"- 语义边：{int(semantic_summary.get('semantic_edge_count') or 0)}",
            f"- API 调用：{int(semantic_summary.get('api_calls') or 0)}",
            f"- 缓存命中 section：{int(semantic_summary.get('cached_sections') or 0)}",
            f"- 总 tokens：{int(semantic_summary.get('total_tokens') or 0)}",
            f"- 耗时：{float(semantic_summary.get('duration_seconds') or 0.0):.3f}s",
            "",
            "## God Nodes",
            "",
        ]
    )
    lines = semantic_lines

    god_node_lines = [
        f"{item.get('source', '')} (degree={item.get('degree', 0)})"
        for item in god_nodes
        if isinstance(item, dict) and item.get("source")
    ]
    lines.extend(_format_report_list(god_node_lines, "暂无显著 hub 节点"))
    lines.extend(["", "## 社区概览", ""])

    if not isinstance(communities, list) or not communities:
        lines.extend(["- 暂无社区数据", ""])
    else:
        for community in communities:
            if not isinstance(community, dict):
                continue
            community_id = str(community.get("id", "") or "community")
            label = str(community.get("label", "") or community_id)
            size = int(community.get("size") or 0)
            kbs = ", ".join(str(item) for item in community.get("kbs", []) if item) or "未分类"
            lines.extend(
                [
                    f"### {community_id}: {label}",
                    "",
                    f"- 文件数：{size}",
                    f"- 知识库：{kbs}",
                    "- Top Files:",
                ]
            )
            top_file_lines = [
                f"{item.get('source', '')} (degree={item.get('degree', 0)})"
                for item in community.get("top_files", [])
                if isinstance(item, dict) and item.get("source")
            ]
            lines.extend(_format_report_list(top_file_lines, "暂无文件摘要"))
            lines.append("- Top Symbols:")
            top_symbol_lines = [
                f"{item.get('name', '')} @ {item.get('source', '')} (score={item.get('score', 0)})"
                for item in community.get("top_symbols", [])
                if isinstance(item, dict) and item.get("name")
            ]
            lines.extend(_format_report_list(top_symbol_lines, "暂无符号摘要"))
            lines.append("- Top Concepts:")
            top_concept_lines = [
                f"{item.get('name', '')} (files={item.get('file_count', 0)}, score={item.get('score', 0)})"
                for item in community.get("top_concepts", [])
                if isinstance(item, dict) and item.get("name")
            ]
            lines.extend(_format_report_list(top_concept_lines, "暂无语义概念"))
            lines.append("- Top Decisions:")
            top_decision_lines = [
                f"{item.get('name', '')} (files={item.get('file_count', 0)}, score={item.get('score', 0)})"
                for item in community.get("top_decisions", [])
                if isinstance(item, dict) and item.get("name")
            ]
            lines.extend(_format_report_list(top_decision_lines, "暂无语义决策"))
            lines.append("- Top Query Notes:")
            top_query_note_lines = [
                f"{item.get('name', '')} (files={item.get('file_count', 0)}, path={item.get('note_relpath', '')})"
                for item in community.get("top_query_notes", [])
                if isinstance(item, dict) and item.get("name")
            ]
            lines.extend(_format_report_list(top_query_note_lines, "暂无知识笔记"))
            lines.append("- Suggested Questions:")
            suggested_query_lines = [
                str(item)
                for item in community.get("suggested_queries", [])
                if isinstance(item, str) and item
            ]
            lines.extend(_format_report_list(suggested_query_lines, "暂无建议问题"))
            lines.append("")

    lines.extend(["## 跨社区连接", ""])
    bridge_lines: list[str] = []
    if isinstance(bridges, list):
        for bridge in bridges:
            if not isinstance(bridge, dict):
                continue
            source_community = str(bridge.get("source_community", "") or "?")
            target_community = str(bridge.get("target_community", "") or "?")
            kind = str(bridge.get("kind", "") or "unknown")
            source = str(bridge.get("source", "") or "")
            target = str(bridge.get("target", "") or "")
            reason = str(bridge.get("reason", "") or "")
            origin = str(bridge.get("origin", "") or "")
            bridge_summary = ""
            raw_bridge_entities = bridge.get("bridges", [])
            if isinstance(raw_bridge_entities, list) and raw_bridge_entities:
                names = [
                    str(item.get("name", "") or item.get("id", ""))
                    for item in raw_bridge_entities
                    if isinstance(item, dict)
                ]
                bridge_summary = f" | bridges: {', '.join(name for name in names if name)}" if names else ""
            bridge_lines.append(
                f"{source_community} -> {target_community} | {kind} | {source} -> {target} | reason: {reason} | origin: {origin}{bridge_summary}"
            )
    lines.extend(_format_report_list(bridge_lines, "暂无跨社区连接"))
    lines.append("")
    return "\n".join(lines)
