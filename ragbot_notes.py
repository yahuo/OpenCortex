from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ragbot import SearchBundle


def _core():
    import ragbot as core

    return core


def _query_notes_dir(persist_path: Path) -> Path:
    core = _core()
    return persist_path / core.WIKI_DIRNAME / core.QUERY_NOTES_DIRNAME


def _extract_markdown_section(text: str, heading: str) -> str:
    pattern = re.compile(
        rf"^## {re.escape(heading)}\s*$\n?(.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return ""
    return match.group(1).strip()


def _unescape_markdown_link_text(text: str) -> str:
    return text.replace("\\\\", "\\").replace("\\[", "[").replace("\\]", "]")


def _parse_query_note_record(note_path: Path, persist_path: Path) -> dict[str, Any] | None:
    core = _core()
    try:
        text = note_path.read_text(encoding="utf-8")
    except OSError:
        return None
    lines = text.splitlines()
    title = ""
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break
    if not title:
        return None

    note_relpath = note_path.relative_to(persist_path).as_posix()
    answer = _extract_markdown_section(text, "结论")
    sources_section = _extract_markdown_section(text, "来源")
    sources: list[str] = []
    for line in sources_section.splitlines():
        match = re.match(r"^- 来源文件：\[(.*)\]\([^)]+\)\s*$", line.strip())
        if not match:
            continue
        sources.append(
            core._normalize_source_path(_unescape_markdown_link_text(match.group(1)))
        )
    sources = core._dedupe_strings(source for source in sources if source)

    created_at = ""
    for line in lines:
        if line.startswith("- 生成时间："):
            created_at = re.sub(r"^- 生成时间：`?(.*?)`?$", r"\1", line.strip())
            break

    tags: list[str] = []
    for line in lines:
        if not line.startswith("- 标签："):
            continue
        tags = re.findall(r"`([^`]+)`", line)
        break

    return {
        "id": core._entity_query_note_node_id(note_relpath),
        "type": "query_note",
        "question": title,
        "name": title,
        "note_relpath": note_relpath,
        "summary": answer[:240].strip(),
        "created_at": created_at,
        "tags": tags,
        "sources": sources,
    }


def _load_query_note_records(persist_path: Path) -> list[dict[str, Any]]:
    query_dir = _query_notes_dir(persist_path)
    if not query_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for note_path in sorted(query_dir.rglob("*.md")):
        record = _parse_query_note_record(note_path, persist_path)
        if record is not None:
            records.append(record)
    return records


def _entity_attachment_files(
    nodes_by_id: dict[str, dict[str, Any]],
    normalized_edges: list[dict[str, Any]],
    attachment_types: set[str] | None = None,
) -> dict[str, set[str]]:
    core = _core()
    selected_types = attachment_types or core.ENTITY_ATTACHMENT_NODE_TYPES
    attachment_node_ids = {
        node_id
        for node_id, node in nodes_by_id.items()
        if str(node.get("type", "") or "") in selected_types
    }
    attachment_files: dict[str, set[str]] = {node_id: set() for node_id in attachment_node_ids}
    for normalized_edge in normalized_edges:
        source_id = str(normalized_edge.get("source", "") or "")
        target_id = str(normalized_edge.get("target", "") or "")
        source_node = nodes_by_id.get(source_id)
        target_node = nodes_by_id.get(target_id)
        if not source_node or not target_node:
            continue
        if source_id in attachment_node_ids:
            target_file = core._entity_node_file_source(target_node)
            if target_file:
                attachment_files.setdefault(source_id, set()).add(target_file)
        if target_id in attachment_node_ids:
            source_file = core._entity_node_file_source(source_node)
            if source_file:
                attachment_files.setdefault(target_id, set()).add(source_file)
    return attachment_files


def _entity_edge_evidence_source(edge: dict[str, Any]) -> str:
    core = _core()
    evidence = edge.get("evidence")
    if not isinstance(evidence, dict):
        return ""
    return core._normalize_source_path(str(evidence.get("source", "") or ""))


def _build_file_only_entity_graph(
    file_sources: list[str],
    document_graph: dict[str, Any],
) -> dict[str, Any]:
    core = _core()
    normalized_sources = sorted(
        {
            core._normalize_source_path(source)
            for source in file_sources
            if core._normalize_source_path(source)
        }
    )
    nodes: list[dict[str, Any]] = []
    node_seen: set[str] = set()
    edges: dict[tuple[Any, ...], dict[str, Any]] = {}
    for source in normalized_sources:
        core._add_entity_node(
            nodes,
            node_seen,
            {
                "id": core._entity_file_node_id(source),
                "type": "file",
                "name": Path(source).name,
                "source": source,
                "path": source,
                "kb": core._extract_kb(source),
                "line_start": 1,
                "line_end": 1,
                "confidence": "EXTRACTED",
            },
        )

    raw_neighbors = document_graph.get("neighbors", {})
    if isinstance(raw_neighbors, dict):
        for source, raw_edges in raw_neighbors.items():
            normalized_source = core._normalize_source_path(source)
            if not normalized_source or not isinstance(raw_edges, list):
                continue
            for edge in raw_edges:
                if not isinstance(edge, dict):
                    continue
                target = core._normalize_source_path(edge.get("target", ""))
                if not target:
                    continue
                core._add_entity_edge(
                    edges,
                    core._entity_file_node_id(normalized_source),
                    core._entity_file_node_id(target),
                    str(edge.get("kind") or ""),
                    evidence_source=normalized_source,
                    reason=str(edge.get("reason") or ""),
                )

    sorted_nodes = sorted(
        nodes,
        key=lambda item: (
            str(item.get("type", "")),
            str(item.get("source", "")),
            str(item.get("name", "")),
        ),
    )
    sorted_edges = sorted(
        edges.values(),
        key=lambda item: (
            str(item.get("source", "")),
            str(item.get("target", "")),
            str(item.get("type", "")),
            str(item.get("reason", "")),
        ),
    )
    return {
        "version": 1,
        "node_count": len(sorted_nodes),
        "edge_count": len(sorted_edges),
        "nodes": sorted_nodes,
        "edges": sorted_edges,
    }


def _normalize_entity_node_id(node_id: str) -> str:
    core = _core()
    text = str(node_id).strip()
    if not text:
        return ""
    if text.startswith("file:"):
        return f"file:{core._normalize_source_path(text[5:])}"
    if text.startswith("section:"):
        body = text[8:]
        source, sep, chunk_index = body.rpartition(":")
        if sep and source:
            return f"section:{core._normalize_source_path(source)}:{chunk_index}"
    if text.startswith("symbol:"):
        body = text[7:]
        parts = body.rsplit(":", 3)
        if len(parts) == 4:
            source, symbol_kind, qualified_name, line_start = parts
            return (
                f"symbol:{core._normalize_source_path(source)}:"
                f"{symbol_kind}:{qualified_name}:{line_start}"
            )
        if len(parts) == 3:
            source, qualified_name, line_start = parts
            return f"symbol:{core._normalize_source_path(source)}:{qualified_name}:{line_start}"
    return text


def _normalize_entity_node_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    core = _core()
    node_id = _normalize_entity_node_id(str(payload.get("id", "")))
    if not node_id:
        return None

    normalized = dict(payload)
    normalized["id"] = node_id
    for key in ("source", "file", "path"):
        value = normalized.get(key)
        if isinstance(value, str):
            clean = core._normalize_source_path(value)
            if clean:
                normalized[key] = clean
            elif key in normalized:
                normalized.pop(key)

    source = str(normalized.get("source", "") or "")
    if source and not normalized.get("kb"):
        normalized["kb"] = core._extract_kb(source)
    return normalized


def _normalize_entity_edge_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    core = _core()
    source = _normalize_entity_node_id(str(payload.get("source", "")))
    target = _normalize_entity_node_id(str(payload.get("target", "")))
    if not source or not target or source == target:
        return None

    normalized = dict(payload)
    normalized["source"] = source
    normalized["target"] = target

    evidence = normalized.get("evidence")
    if isinstance(evidence, dict):
        normalized_evidence = dict(evidence)
        evidence_source = normalized_evidence.get("source")
        if isinstance(evidence_source, str):
            clean = core._normalize_source_path(evidence_source)
            if clean:
                normalized_evidence["source"] = clean
            else:
                normalized_evidence.pop("source", None)
        normalized["evidence"] = normalized_evidence
    return normalized


def _merge_query_notes_into_entity_graph(
    entity_graph: dict[str, Any],
    query_notes: list[dict[str, Any]],
) -> dict[str, Any]:
    core = _core()
    nodes: list[dict[str, Any]] = []
    node_seen: set[str] = set()
    nodes_by_id: dict[str, dict[str, Any]] = {}
    query_note_ids: set[str] = set()
    for raw_node in entity_graph.get("nodes", []):
        if not isinstance(raw_node, dict):
            continue
        normalized_node = _normalize_entity_node_payload(raw_node)
        if normalized_node is None:
            continue
        if str(normalized_node.get("type", "") or "") == "query_note":
            query_note_ids.add(str(normalized_node["id"]))
            continue
        core._add_entity_node(nodes, node_seen, normalized_node)
        nodes_by_id[str(normalized_node["id"])] = normalized_node

    edges: dict[tuple[Any, ...], dict[str, Any]] = {}
    normalized_edges: list[dict[str, Any]] = []
    for raw_edge in entity_graph.get("edges", []):
        if not isinstance(raw_edge, dict):
            continue
        normalized_edge = _normalize_entity_edge_payload(raw_edge)
        if normalized_edge is None:
            continue
        source_id = str(normalized_edge.get("source", "") or "")
        target_id = str(normalized_edge.get("target", "") or "")
        if source_id in query_note_ids or target_id in query_note_ids:
            continue
        normalized_edges.append(normalized_edge)
        evidence = normalized_edge.get("evidence")
        evidence_source = ""
        line_start = None
        line_end = None
        if isinstance(evidence, dict):
            evidence_source = core._normalize_source_path(str(evidence.get("source", "") or ""))
            line_start = evidence.get("line_start")
            line_end = evidence.get("line_end")
        if not evidence_source:
            evidence_source = core._normalize_source_path(
                str(normalized_edge.get("reason", "") or "")
            )
        if not evidence_source:
            evidence_source = "graph"
        core._add_entity_edge(
            edges,
            source_id=source_id,
            target_id=target_id,
            kind=str(normalized_edge.get("type", "") or ""),
            evidence_source=evidence_source,
            reason=str(normalized_edge.get("reason", "") or ""),
            line_start=line_start if isinstance(line_start, int) else None,
            line_end=line_end if isinstance(line_end, int) else None,
        )

    semantic_attachment_files = _entity_attachment_files(
        nodes_by_id,
        normalized_edges,
        attachment_types={"concept", "decision"},
    )

    for record in query_notes:
        note_id = str(record.get("id", "") or "").strip()
        question = str(record.get("question", "") or "").strip()
        note_relpath = core._normalize_source_path(str(record.get("note_relpath", "") or ""))
        if not note_id or not question or not note_relpath:
            continue
        payload = {
            "id": note_id,
            "type": "query_note",
            "name": question,
            "summary": str(record.get("summary", "") or "").strip(),
            "note_relpath": note_relpath,
            "created_at": str(record.get("created_at", "") or ""),
            "tags": [str(tag) for tag in record.get("tags", []) if str(tag).strip()],
            "confidence": "EXTRACTED",
        }
        core._add_entity_node(nodes, node_seen, payload)
        nodes_by_id[note_id] = payload
        for source in record.get("sources", []):
            normalized_source = core._normalize_source_path(source)
            file_node_id = core._entity_file_node_id(normalized_source)
            if file_node_id not in nodes_by_id:
                continue
            core._add_entity_edge(
                edges,
                source_id=file_node_id,
                target_id=note_id,
                kind="semantically_related",
                evidence_source=note_relpath,
                reason=question,
            )
            core._add_entity_edge(
                edges,
                source_id=note_id,
                target_id=file_node_id,
                kind="semantically_related",
                evidence_source=note_relpath,
                reason=question,
            )
            for semantic_node_id, attached_files in semantic_attachment_files.items():
                if normalized_source not in attached_files:
                    continue
                semantic_node = nodes_by_id.get(semantic_node_id)
                if semantic_node is None:
                    continue
                relation = str(semantic_node.get("type", "") or "semantically_related")
                core._add_entity_edge(
                    edges,
                    source_id=note_id,
                    target_id=semantic_node_id,
                    kind="semantically_related",
                    evidence_source=normalized_source,
                    reason=question or relation,
                )
                core._add_entity_edge(
                    edges,
                    source_id=semantic_node_id,
                    target_id=note_id,
                    kind="semantically_related",
                    evidence_source=normalized_source,
                    reason=question or relation,
                )

    sorted_nodes = sorted(
        nodes,
        key=lambda item: (
            str(item.get("type", "")),
            str(item.get("source", "")),
            str(item.get("note_relpath", "")),
            int(item.get("chunk_index") or 0),
            int(item.get("line_start") or 0),
            str(item.get("qualified_name") or item.get("name") or ""),
        ),
    )
    sorted_edges = sorted(
        edges.values(),
        key=lambda item: (
            str(item.get("source", "")),
            str(item.get("target", "")),
            str(item.get("type", "")),
            str(item.get("reason", "")),
        ),
    )
    return {
        **entity_graph,
        "nodes": sorted_nodes,
        "edges": sorted_edges,
        "node_count": len(sorted_nodes),
        "edge_count": len(sorted_edges),
    }


def refresh_query_note_graph_artifacts(persist_path: Path) -> None:
    core = _core()
    manifest_path = persist_path / "index_manifest.json"
    document_graph_path = persist_path / core.DOCUMENT_GRAPH_FILENAME
    entity_graph_path = persist_path / core.ENTITY_GRAPH_FILENAME
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(manifest, dict):
        return

    document_graph: dict[str, Any] = {"version": 1, "edge_count": 0, "neighbors": {}}
    if document_graph_path.exists():
        try:
            loaded_document_graph = json.loads(document_graph_path.read_text(encoding="utf-8"))
            if isinstance(loaded_document_graph, dict):
                document_graph = loaded_document_graph
        except (OSError, json.JSONDecodeError):
            document_graph = {"version": 1, "edge_count": 0, "neighbors": {}}

    file_sources = [
        core._normalize_source_path(str(entry.get("name", "") or ""))
        for entry in manifest.get("files", [])
        if isinstance(entry, dict)
    ]

    entity_graph: dict[str, Any]
    if entity_graph_path.exists():
        try:
            loaded_entity_graph = json.loads(entity_graph_path.read_text(encoding="utf-8"))
            entity_graph = (
                loaded_entity_graph
                if isinstance(loaded_entity_graph, dict)
                else _build_file_only_entity_graph(file_sources, document_graph)
            )
        except (OSError, json.JSONDecodeError):
            entity_graph = _build_file_only_entity_graph(file_sources, document_graph)
    else:
        entity_graph = _build_file_only_entity_graph(file_sources, document_graph)

    entity_graph = _merge_query_notes_into_entity_graph(
        entity_graph,
        _load_query_note_records(persist_path),
    )
    entity_graph_path.write_text(
        json.dumps(entity_graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    community_index = core._build_community_index(
        file_sources=file_sources,
        document_graph=document_graph,
        entity_graph=entity_graph,
        semantic_stats=manifest.get("semantic_graph_stats", {}),
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

