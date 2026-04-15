from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from langchain_community.vectorstores import FAISS


def _core():
    import ragbot as core

    return core


@dataclass(slots=True)
class SearchBundle:
    vectorstore: FAISS
    persist_dir: Path
    source_dir: Path | None
    manifest: dict[str, Any]
    files: list[dict[str, Any]]
    files_by_source: dict[str, dict[str, Any]]
    normalized_text_dir: Path
    symbol_index: list[dict[str, Any]]
    document_graph: dict[str, Any]
    graph_neighbors: dict[str, list[dict[str, Any]]]
    entity_graph: dict[str, Any]
    entity_nodes_by_id: dict[str, dict[str, Any]]
    entity_edges_by_source: dict[str, list[dict[str, Any]]]
    wiki_pages: list[dict[str, Any]]
    fulltext_index: dict[str, Any] = field(default_factory=dict)
    query_expansion_index: dict[str, Any] = field(default_factory=dict)

    def cache_path_for(self, source: str) -> Path | None:
        entry = self.files_by_source.get(source)
        if not entry:
            return None
        relative = entry.get("normalized_text")
        if not relative:
            return None
        return self.normalized_text_dir / relative


def _runtime_wiki_artifact_paths(persist_path: Path) -> list[str]:
    core = _core()
    wiki_dir = persist_path / core.WIKI_DIRNAME
    if not wiki_dir.exists():
        return []

    paths: list[str] = []
    for page_path in sorted(wiki_dir.rglob("*.md")):
        if not page_path.is_file():
            continue
        relative = page_path.relative_to(persist_path).as_posix()
        if relative == f"{core.WIKI_DIRNAME}/log.md":
            continue
        paths.append(relative)
    return paths


def load_fulltext_index_payload(fulltext_index_path: Path) -> dict[str, Any]:
    if not fulltext_index_path.exists():
        raise ValueError(f"缺少全文索引文件：{fulltext_index_path.name}")
    try:
        payload = json.loads(fulltext_index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"全文索引文件损坏：{fulltext_index_path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"全文索引格式无效：{fulltext_index_path.name}")

    raw_chunks = payload.get("chunks")
    raw_postings = payload.get("postings")
    if not isinstance(raw_chunks, list) or not isinstance(raw_postings, dict):
        raise ValueError(f"全文索引格式无效：{fulltext_index_path.name}")
    return payload


def search_bundle_artifact_signature(
    persist_dir: str | Path,
) -> tuple[tuple[str, bool, int, int], ...]:
    core = _core()
    persist_path = Path(persist_dir)
    manifest_path = persist_path / "index_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded_manifest, dict):
                manifest = loaded_manifest
        except (OSError, json.JSONDecodeError):
            manifest = {}

    relative_paths = [
        "index.faiss",
        "index.pkl",
        "index_manifest.json",
        str(manifest.get("symbol_index_file", core.SYMBOL_INDEX_FILENAME) or core.SYMBOL_INDEX_FILENAME),
        str(
            manifest.get("document_graph_file", core.DOCUMENT_GRAPH_FILENAME)
            or core.DOCUMENT_GRAPH_FILENAME
        ),
        str(manifest.get("entity_graph_file", core.ENTITY_GRAPH_FILENAME) or core.ENTITY_GRAPH_FILENAME),
        str(
            manifest.get("fulltext_index_file", core.FULLTEXT_INDEX_FILENAME)
            or core.FULLTEXT_INDEX_FILENAME
        ),
        str(
            manifest.get("community_index_file", core.COMMUNITY_INDEX_FILENAME)
            or core.COMMUNITY_INDEX_FILENAME
        ),
        str(manifest.get("lint_report_file", core.LINT_REPORT_FILENAME) or core.LINT_REPORT_FILENAME),
        str(
            manifest.get("graph_report_file", f"{core.REPORTS_DIRNAME}/{core.GRAPH_REPORT_FILENAME}")
            or f"{core.REPORTS_DIRNAME}/{core.GRAPH_REPORT_FILENAME}"
        ),
    ]
    relative_paths.extend(_runtime_wiki_artifact_paths(persist_path))

    signature: list[tuple[str, bool, int, int]] = []
    seen: set[str] = set()
    for relative_path in relative_paths:
        normalized = Path(relative_path).as_posix()
        if normalized in seen:
            continue
        seen.add(normalized)
        artifact_path = persist_path / normalized
        try:
            stat = artifact_path.stat()
        except OSError:
            signature.append((normalized, False, 0, 0))
            continue
        signature.append((normalized, True, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def _markdown_heading_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return fallback


def _source_from_wiki_file_relpath(relpath: str) -> str:
    core = _core()
    normalized = core._normalize_source_path(relpath)
    if not normalized.startswith("files/") or not normalized.endswith(".md"):
        return ""
    return core._normalize_source_path(normalized[len("files/") : -3])


def _wiki_page_kind(relpath: str) -> str:
    core = _core()
    normalized = core._normalize_source_path(relpath)
    if normalized == "index.md":
        return "index"
    if normalized == "log.md":
        return "log"
    if normalized.startswith("files/"):
        return "file"
    if normalized.startswith(f"{core.QUERY_NOTES_DIRNAME}/"):
        return "query"
    if normalized.startswith("communities/"):
        return "community"
    if normalized.startswith("entities/"):
        return "entity"
    return ""


def _source_from_wiki_link_target(target: str) -> str:
    cleaned = unquote(str(target).strip()).split("#", 1)[0].split("?", 1)[0]
    if cleaned.startswith("../"):
        cleaned = cleaned[3:]
    return _source_from_wiki_file_relpath(cleaned)


def _iter_wiki_markdown_links(text: str) -> list[str]:
    core = _core()
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
            for target in core.MARKDOWN_LINK_RE.findall(line)
            if isinstance(target, str) and target.strip()
        )
    return links


def _load_wiki_pages(persist_dir: Path, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    core = _core()
    wiki_dir = persist_dir / core.WIKI_DIRNAME
    if not wiki_dir.exists():
        return []

    file_relpaths = {
        core._normalize_source_path(
            (Path("files") / Path(f"{str(entry.get('name', '') or '')}.md")).as_posix()
        ): core._normalize_source_path(str(entry.get("name", "") or ""))
        for entry in files
        if core._normalize_source_path(str(entry.get("name", "") or ""))
    }
    from wiki import is_wiki_write_in_progress

    last_error: OSError | None = None
    for attempt in range(3):
        pages: list[dict[str, Any]] = []
        try:
            for page_path in sorted(wiki_dir.rglob("*.md")):
                relpath = core._normalize_source_path(page_path.relative_to(wiki_dir).as_posix())
                kind = _wiki_page_kind(relpath)
                if not kind or kind == "log":
                    continue
                text = page_path.read_text(encoding="utf-8")
                if kind == "file":
                    source = file_relpaths.get(relpath) or _source_from_wiki_file_relpath(relpath)
                    source_refs = [source] if source else []
                    title = source or _markdown_heading_title(text, page_path.stem)
                    page_id = f"wiki:file:{source or relpath}"
                else:
                    source_refs = core._dedupe_strings(
                        _source_from_wiki_link_target(target)
                        for target in _iter_wiki_markdown_links(text)
                    )
                    title = "知识导航" if kind == "index" else _markdown_heading_title(text, page_path.stem)
                    page_id = f"wiki:{kind}:{relpath}"
                pages.append(
                    {
                        "id": page_id,
                        "kind": kind,
                        "title": title,
                        "relpath": relpath,
                        "text": text,
                        "source_refs": source_refs,
                    }
                )
            return pages
        except OSError as exc:
            last_error = exc
            if is_wiki_write_in_progress(persist_dir) and attempt < 2:
                core.time.sleep(0.05)
                continue
            raise

    if last_error is not None:
        raise last_error
    return []


def load_search_bundle(
    embed_api_key: str,
    embed_base_url: str | None = None,
    embed_model: str | None = None,
    persist_dir: str | Path | None = None,
) -> SearchBundle | None:
    core = _core()
    resolved_base_url = core.SILICONFLOW_BASE_URL if embed_base_url is None else embed_base_url
    resolved_embed_model = core.DEFAULT_EMBED_MODEL if embed_model is None else embed_model
    resolved_persist_dir = core.DEFAULT_FAISS_DIR if persist_dir is None else str(persist_dir)

    index_file = Path(resolved_persist_dir) / "index.faiss"
    if not index_file.exists():
        return None

    embeddings = core.make_embeddings(
        api_key=embed_api_key,
        base_url=resolved_base_url,
        model=resolved_embed_model,
    )
    vectorstore = FAISS.load_local(
        str(resolved_persist_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    manifest_path = Path(resolved_persist_dir) / "index_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"files": []}
    )
    raw_files = manifest.get("files", [])
    files: list[dict[str, Any]] = []
    for entry in raw_files if isinstance(raw_files, list) else []:
        if not isinstance(entry, dict):
            continue
        normalized_entry = dict(entry)
        source = core._normalize_source_path(normalized_entry.get("name", ""))
        if not source:
            continue
        normalized_entry["name"] = source
        normalized_entry["kb"] = core._extract_kb(source) or normalized_entry.get("kb", "")
        normalized_text = normalized_entry.get("normalized_text", "")
        if isinstance(normalized_text, str):
            normalized_entry["normalized_text"] = core._normalize_source_path(normalized_text)
        files.append(normalized_entry)
    manifest["files"] = files
    files_by_source = {entry.get("name", ""): entry for entry in files if entry.get("name")}
    normalized_text_dir = Path(resolved_persist_dir) / manifest.get(
        "normalized_text_dir",
        core.NORMALIZED_TEXT_DIRNAME,
    )
    symbol_index_path = Path(resolved_persist_dir) / manifest.get(
        "symbol_index_file",
        core.SYMBOL_INDEX_FILENAME,
    )
    document_graph_path = Path(resolved_persist_dir) / manifest.get(
        "document_graph_file",
        core.DOCUMENT_GRAPH_FILENAME,
    )
    entity_graph_path = Path(resolved_persist_dir) / manifest.get(
        "entity_graph_file",
        core.ENTITY_GRAPH_FILENAME,
    )
    fulltext_index_path = Path(resolved_persist_dir) / manifest.get(
        "fulltext_index_file",
        core.FULLTEXT_INDEX_FILENAME,
    )

    symbol_index: list[dict[str, Any]] = []
    if symbol_index_path.exists():
        for line in symbol_index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            source = core._normalize_source_path(record.get("source", ""))
            if not source:
                continue
            record["source"] = source
            symbol_index.append(record)

    document_graph: dict[str, Any] = {"version": 1, "edge_count": 0, "neighbors": {}}
    if document_graph_path.exists():
        try:
            document_graph = json.loads(document_graph_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            document_graph = {"version": 1, "edge_count": 0, "neighbors": {}}
    graph_neighbors: dict[str, list[dict[str, Any]]] = {}
    raw_neighbors = document_graph.get("neighbors", {})
    if isinstance(raw_neighbors, dict):
        for source, edges in raw_neighbors.items():
            normalized_source = core._normalize_source_path(source)
            if not normalized_source or not isinstance(edges, list):
                continue
            bucket = graph_neighbors.setdefault(normalized_source, [])
            seen_edges: set[tuple[str, str, str, int]] = set()
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                target = core._normalize_source_path(edge.get("target", ""))
                if not target or target == normalized_source:
                    continue
                payload = dict(edge)
                payload["target"] = target
                key = (
                    str(payload.get("kind", "")),
                    target,
                    str(payload.get("reason", "")),
                    int(payload.get("hop") or 0),
                )
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                bucket.append(payload)
    document_graph = {
        **document_graph,
        "neighbors": graph_neighbors,
        "edge_count": sum(len(edges) for edges in graph_neighbors.values()),
    }

    raw_entity_graph: dict[str, Any] = {
        "version": 1,
        "node_count": 0,
        "edge_count": 0,
        "nodes": [],
        "edges": [],
    }
    if entity_graph_path.exists():
        try:
            raw_entity_graph = json.loads(entity_graph_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw_entity_graph = {
                "version": 1,
                "node_count": 0,
                "edge_count": 0,
                "nodes": [],
                "edges": [],
            }

    entity_nodes: list[dict[str, Any]] = []
    entity_nodes_by_id: dict[str, dict[str, Any]] = {}
    raw_nodes = raw_entity_graph.get("nodes", [])
    if isinstance(raw_nodes, list):
        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict):
                continue
            normalized_node = core._normalize_entity_node_payload(raw_node)
            if normalized_node is None:
                continue
            node_id = str(normalized_node["id"])
            if node_id in entity_nodes_by_id:
                continue
            entity_nodes_by_id[node_id] = normalized_node
            entity_nodes.append(normalized_node)

    entity_edges: list[dict[str, Any]] = []
    entity_edges_by_source: dict[str, list[dict[str, Any]]] = {}
    seen_entity_edges: set[tuple[Any, ...]] = set()
    raw_edges = raw_entity_graph.get("edges", [])
    if isinstance(raw_edges, list):
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                continue
            normalized_edge = core._normalize_entity_edge_payload(raw_edge)
            if normalized_edge is None:
                continue
            evidence = normalized_edge.get("evidence")
            evidence_source = ""
            evidence_line_start = None
            evidence_line_end = None
            if isinstance(evidence, dict):
                evidence_source = str(evidence.get("source", "") or "")
                evidence_line_start = evidence.get("line_start")
                evidence_line_end = evidence.get("line_end")
            dedupe_key = (
                normalized_edge["source"],
                normalized_edge["target"],
                normalized_edge.get("type", ""),
                normalized_edge.get("reason", ""),
                evidence_source,
                evidence_line_start,
                evidence_line_end,
            )
            if dedupe_key in seen_entity_edges:
                continue
            seen_entity_edges.add(dedupe_key)
            entity_edges.append(normalized_edge)
            entity_edges_by_source.setdefault(str(normalized_edge["source"]), []).append(normalized_edge)

    entity_graph = {
        **raw_entity_graph,
        "nodes": entity_nodes,
        "edges": entity_edges,
        "node_count": len(entity_nodes),
        "edge_count": len(entity_edges),
    }

    fulltext_index: dict[str, Any] = {
        "version": 1,
        "doc_count": 0,
        "avg_chunk_length": 0.0,
        "chunks": {},
        "postings": {},
    }
    raw_fulltext_index = load_fulltext_index_payload(fulltext_index_path)

    chunks_by_id: dict[int, dict[str, Any]] = {}
    raw_chunks = raw_fulltext_index["chunks"]
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            continue
        try:
            chunk_id = int(raw_chunk.get("id"))
        except (TypeError, ValueError):
            continue
        source = core._normalize_source_path(raw_chunk.get("source", ""))
        if not source:
            continue
        chunks_by_id[chunk_id] = {
            "id": chunk_id,
            "source": source,
            "chunk_index": int(raw_chunk.get("chunk_index") or 0),
            "line_start": raw_chunk.get("line_start"),
            "line_end": raw_chunk.get("line_end"),
            "label": str(raw_chunk.get("label", "") or ""),
            "length": max(1, int(raw_chunk.get("length") or 1)),
        }
    postings: dict[str, list[tuple[int, int]]] = {}
    raw_postings = raw_fulltext_index["postings"]
    for raw_term, raw_entries in raw_postings.items():
        term = str(raw_term).strip().lower()
        if not term or not isinstance(raw_entries, list):
            continue
        normalized_entries: list[tuple[int, int]] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, list | tuple) or len(raw_entry) < 2:
                continue
            try:
                chunk_id = int(raw_entry[0])
                tf = int(raw_entry[1])
            except (TypeError, ValueError):
                continue
            if tf <= 0 or chunk_id not in chunks_by_id:
                continue
            normalized_entries.append((chunk_id, tf))
        if normalized_entries:
            postings[term] = normalized_entries
    doc_count = int(raw_fulltext_index.get("doc_count") or len(chunks_by_id))
    if doc_count <= 0:
        doc_count = len(chunks_by_id)
    avg_chunk_length = float(raw_fulltext_index.get("avg_chunk_length") or 0.0)
    if avg_chunk_length <= 0 and chunks_by_id:
        avg_chunk_length = sum(
            int(chunk.get("length") or 1) for chunk in chunks_by_id.values()
        ) / len(chunks_by_id)
    fulltext_index = {
        "version": int(raw_fulltext_index.get("version") or 1),
        "doc_count": doc_count,
        "avg_chunk_length": avg_chunk_length,
        "chunks": chunks_by_id,
        "postings": postings,
    }

    source_dir = manifest.get("source_dir")
    wiki_pages = _load_wiki_pages(Path(resolved_persist_dir), files)
    query_expansion_index = core._build_query_expansion_index(
        files=files,
        symbol_index=symbol_index,
        entity_nodes=entity_nodes,
    )
    return SearchBundle(
        vectorstore=vectorstore,
        persist_dir=Path(resolved_persist_dir),
        source_dir=Path(source_dir) if source_dir else None,
        manifest=manifest,
        files=files,
        files_by_source=files_by_source,
        normalized_text_dir=normalized_text_dir,
        symbol_index=symbol_index,
        document_graph=document_graph,
        graph_neighbors=graph_neighbors,
        entity_graph=entity_graph,
        entity_nodes_by_id=entity_nodes_by_id,
        entity_edges_by_source=entity_edges_by_source,
        wiki_pages=wiki_pages,
        fulltext_index=fulltext_index,
        query_expansion_index=query_expansion_index,
    )


def load_vectorstore(
    embed_api_key: str,
    embed_base_url: str | None = None,
    embed_model: str | None = None,
    persist_dir: str | Path | None = None,
) -> FAISS | None:
    bundle = load_search_bundle(
        embed_api_key=embed_api_key,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        persist_dir=persist_dir,
    )
    return bundle.vectorstore if bundle else None


__all__ = [
    "SearchBundle",
    "_load_wiki_pages",
    "load_search_bundle",
    "load_vectorstore",
    "search_bundle_artifact_signature",
]
