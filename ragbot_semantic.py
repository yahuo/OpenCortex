from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
import posixpath
import re
from threading import local
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from ragbot_artifacts import atomic_write_text

if TYPE_CHECKING:
    from ragbot import IndexedFile


def _core():
    import ragbot as core

    return core


def _entity_semantic_node_id(node_type: str, name: str) -> str:
    normalized_type = str(node_type).strip().lower() or "concept"
    normalized_raw_name = name.strip().lower()
    normalized_name = re.sub(r"[^a-z0-9]+", "-", normalized_raw_name).strip("-")
    slug = normalized_name or "node"
    digest = hashlib.sha1(normalized_raw_name.encode("utf-8")).hexdigest()[:12]
    return f"{normalized_type}:{slug}:{digest}"


def _extract_section_reference_tokens(text: str) -> list[str]:
    core = _core()
    seen: set[str] = set()
    tokens: list[str] = []
    for raw in core.CODE_SPAN_RE.findall(text):
        token = raw.strip()
        lowered = token.lower()
        if token and lowered not in seen:
            seen.add(lowered)
            tokens.append(token)
    for raw in core.CALL_RE.findall(text):
        token = raw.strip()
        lowered = token.lower()
        if token and lowered not in seen:
            seen.add(lowered)
            tokens.append(token)
    return tokens


def _build_symbol_reference_lookup(
    symbol_entries: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    seen_keys: set[tuple[str, str]] = set()
    for entry in symbol_entries:
        for raw in (entry.get("name", ""), entry.get("qualified_name", "")):
            key = str(raw).strip().lower()
            if not key:
                continue
            dedupe_key = (key, str(entry.get("id", "")))
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            lookup.setdefault(key, []).append(entry)
    return lookup


def _resolve_symbol_reference(
    source: str,
    token: str,
    lookup: dict[str, list[dict[str, Any]]],
) -> str | None:
    core = _core()
    key = token.strip().lower()
    if not key:
        return None

    candidates = lookup.get(key, [])
    if not candidates:
        return None

    source = core._normalize_source_path(source)
    source_kb = core._extract_kb(source)

    def _prefer_specific(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        non_module = [entry for entry in entries if entry.get("symbol_kind") != "module"]
        return non_module or entries

    same_file = _prefer_specific([entry for entry in candidates if entry.get("source") == source])
    if len(same_file) == 1:
        return str(same_file[0]["id"])

    same_kb = _prefer_specific([entry for entry in candidates if entry.get("kb") == source_kb])
    if len(same_kb) == 1:
        return str(same_kb[0]["id"])

    candidates = _prefer_specific(candidates)
    if len(candidates) == 1:
        return str(candidates[0]["id"])
    return None


def _build_python_module_lookup(indexed_files: list[IndexedFile]) -> dict[str, list[str]]:
    core = _core()
    lookup: dict[str, list[str]] = {}
    for indexed in indexed_files:
        if indexed.suffix not in core.PYTHON_SUFFIXES:
            continue
        source = core._normalize_source_path(indexed.rel_path)
        if not source:
            continue
        module_path = str(Path(source).with_suffix("")).replace("\\", "/")
        keys = {module_path.replace("/", "."), Path(source).stem}
        if Path(source).name == "__init__.py":
            package_name = posixpath.dirname(module_path).replace("/", ".")
            if package_name:
                keys.add(package_name)
        for key in keys:
            lowered = key.strip().lower()
            if not lowered:
                continue
            lookup.setdefault(lowered, []).append(source)
    return lookup


def _resolve_import_reference(
    source: str,
    qualified_name: str,
    module_lookup: dict[str, list[str]],
) -> str | None:
    core = _core()
    qualified_name = qualified_name.strip().lower()
    if not qualified_name:
        return None

    source_kb = core._extract_kb(source)
    parts = qualified_name.split(".")
    for end in range(len(parts), 0, -1):
        candidate = ".".join(parts[:end])
        matches = module_lookup.get(candidate, [])
        if not matches:
            continue
        same_kb = [match for match in matches if core._extract_kb(match) == source_kb]
        if len(same_kb) == 1:
            return same_kb[0]
        if len(matches) == 1:
            return matches[0]
    return None


def _semantic_graph_enabled(
    llm_api_key: str,
    llm_model: str,
    llm_base_url: str,
) -> tuple[bool, str]:
    core = _core()
    raw = os.getenv("SEMANTIC_GRAPH_ENABLED", "auto").strip().lower()
    enabled = bool(llm_api_key.strip() and llm_model.strip() and llm_base_url.strip())
    if raw in core.SEMANTIC_GRAPH_DISABLED_VALUES:
        return False, "disabled_by_env"
    if raw in core.SEMANTIC_GRAPH_ENABLED_VALUES:
        if enabled:
            return True, ""
        return False, "missing_llm_config"
    if enabled:
        return True, ""
    return False, "missing_llm_config"


def _semantic_section_fingerprint(
    source: str,
    chunk_index: int,
    text: str,
    llm_model: str,
    llm_base_url: str,
) -> str:
    core = _core()
    digest = hashlib.sha256()
    digest.update(source.encode("utf-8"))
    digest.update(f":{chunk_index}:".encode("utf-8"))
    digest.update(llm_model.encode("utf-8"))
    digest.update(llm_base_url.strip().rstrip("/").encode("utf-8"))
    digest.update(core.SEMANTIC_PROMPT_VERSION.encode("utf-8"))
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _semantic_cache_path(persist_path: Path) -> Path:
    core = _core()
    return persist_path / core.SEMANTIC_EXTRACT_CACHE_FILENAME


def _load_semantic_cache(persist_path: Path) -> dict[str, Any]:
    path = _semantic_cache_path(persist_path)
    if not path.exists():
        return {"version": 1, "entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "entries": {}}
    if not isinstance(payload, dict):
        return {"version": 1, "entries": {}}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        payload["entries"] = {}
    return payload


def _write_semantic_cache(persist_path: Path, cache_payload: dict[str, Any]) -> None:
    atomic_write_text(
        _semantic_cache_path(persist_path),
        json.dumps(cache_payload, ensure_ascii=False, indent=2),
    )


def _semantic_cache_key(section: dict[str, Any]) -> str:
    core = _core()
    return core._entity_section_node_id(str(section["source"]), int(section["chunk_index"]))


def _semantic_progress_message(processed: int, total: int, stats: dict[str, Any]) -> str:
    return (
        f"语义抽取中 {processed}/{total}"
        f"（缓存 {int(stats.get('cached_sections') or 0)}"
        f" / 新抽取 {int(stats.get('extracted_sections') or 0)}"
        f" / 失败 {int(stats.get('failed_sections') or 0)}）..."
    )


def _semantic_aliases(name: str, aliases: Any) -> list[str]:
    core = _core()
    values = [name]
    if isinstance(aliases, list):
        values.extend(str(item).strip() for item in aliases if str(item).strip())
    return core._dedupe_strings(values)


def _register_semantic_aliases(
    alias_lookup: dict[str, list[str]],
    aliases: Iterable[str],
    node_id: str,
) -> None:
    for alias in aliases:
        lowered = str(alias).strip().lower()
        if not lowered:
            continue
        bucket = alias_lookup.setdefault(lowered, [])
        if node_id not in bucket:
            bucket.append(node_id)


def _semantic_extraction_prompt(section: dict[str, Any]) -> str:
    core = _core()
    text = str(section.get("text", "") or "").strip()
    if len(text) > core.SEMANTIC_SECTION_MAX_CHARS:
        text = text[: core.SEMANTIC_SECTION_MAX_CHARS - 3].rstrip() + "..."
    source = str(section.get("source", "") or "")
    label = str(section.get("label", "") or "")
    return f"""你是知识图谱抽取器。请只根据给定片段抽取明确出现或被明确陈述的语义概念和决策。

只返回一个 JSON 对象，不要输出任何额外说明，结构必须是：
{{
  "concepts": [
    {{
      "name": "概念名",
      "summary": "一句话说明",
      "aliases": ["可选别名"]
    }}
  ],
  "decisions": [
    {{
      "name": "决策名",
      "summary": "一句话说明",
      "aliases": ["可选别名"],
      "rationale": ["支撑该决策的概念、理由或依据短语"]
    }}
  ]
}}

要求：
1. 只有文本里明确表达的抽象概念、业务术语、系统名称、架构决策才能抽取。
2. `summary` 必须简短，不要重复原文大段句子。
3. 如果没有可抽取内容，返回空数组。
4. `rationale` 只保留能直接从文本看出的依据短语。

片段来源：{source}
片段标签：{label}

片段正文：
\"\"\"
{text}
\"\"\"
"""


def _response_token_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        response_metadata = getattr(response, "response_metadata", None)
        if isinstance(response_metadata, dict):
            raw_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
            if isinstance(raw_usage, dict):
                usage = raw_usage
    if not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    prompt_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _normalize_semantic_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    name = str(item.get("name", "") or "").strip()
    if not name:
        return None
    aliases = _semantic_aliases(name, item.get("aliases"))
    payload: dict[str, Any] = {
        "name": name,
        "aliases": aliases,
    }
    summary = str(item.get("summary", "") or "").strip()
    if summary:
        payload["summary"] = summary
    rationale = item.get("rationale")
    if isinstance(rationale, list):
        core = _core()
        payload["rationale"] = core._dedupe_strings(str(entry) for entry in rationale if str(entry).strip())
    return payload


def _parse_semantic_payload(content: str) -> dict[str, list[dict[str, Any]]]:
    core = _core()
    payload = core._extract_json_blob(content)
    concepts = payload.get("concepts") or []
    decisions = payload.get("decisions") or []
    return {
        "concepts": [item for item in (_normalize_semantic_item(entry) for entry in concepts) if item],
        "decisions": [item for item in (_normalize_semantic_item(entry) for entry in decisions) if item],
    }


def _upsert_semantic_node(
    nodes_by_id: dict[str, dict[str, Any]],
    nodes: list[dict[str, Any]],
    node_seen: set[str],
    alias_lookup: dict[str, list[str]],
    node_type: str,
    item: dict[str, Any],
) -> str:
    core = _core()
    name = str(item.get("name", "") or "").strip()
    node_id = _entity_semantic_node_id(node_type, name)
    aliases = _semantic_aliases(name, item.get("aliases"))
    if node_id not in nodes_by_id:
        payload = {
            "id": node_id,
            "type": node_type,
            "name": name,
            "aliases": aliases,
            "summary": str(item.get("summary", "") or "").strip(),
            "confidence": "INFERRED",
        }
        core._add_entity_node(nodes, node_seen, payload)
        nodes_by_id[node_id] = payload
    else:
        payload = nodes_by_id[node_id]
        existing_aliases = payload.get("aliases", [])
        merged_aliases = core._dedupe_strings(existing_aliases if isinstance(existing_aliases, list) else [])
        lowered_existing = {entry.lower() for entry in merged_aliases}
        for alias in aliases:
            if alias.lower() not in lowered_existing:
                merged_aliases.append(alias)
                lowered_existing.add(alias.lower())
        payload["aliases"] = merged_aliases
        if not str(payload.get("summary", "") or "").strip():
            summary = str(item.get("summary", "") or "").strip()
            if summary:
                payload["summary"] = summary
    _register_semantic_aliases(alias_lookup, aliases, node_id)
    return node_id


def _iter_semantic_sections(indexed_files: list[IndexedFile]) -> Iterable[dict[str, Any]]:
    core = _core()
    for indexed in indexed_files:
        source = core._normalize_source_path(indexed.rel_path)
        if not source:
            continue
        for chunk_index, chunk in enumerate(indexed.iter_chunks()):
            text = chunk.text.strip()
            if not text:
                continue
            yield {
                "source": source,
                "kb": indexed.kb,
                "suffix": indexed.suffix,
                "chunk_index": chunk_index,
                "label": core._chunk_location_label(chunk, f"chunk {chunk_index + 1}"),
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "text": text,
            }


def _extract_semantic_sections(
    indexed_files: list[IndexedFile],
    persist_path: Path,
    llm_api_key: str,
    llm_model: str,
    llm_base_url: str,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    core = _core()
    normalized_llm_base_url = llm_base_url.strip().rstrip("/")
    enabled, disabled_reason = _semantic_graph_enabled(
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
    )
    stats: dict[str, Any] = {
        "enabled": enabled,
        "disabled_reason": disabled_reason,
        "llm_model": llm_model,
        "prompt_version": core.SEMANTIC_PROMPT_VERSION,
        "sections_total": 0,
        "cached_sections": 0,
        "extracted_sections": 0,
        "failed_sections": 0,
        "api_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "duration_seconds": 0.0,
        "concurrency": 1,
        "cache_flushes": 0,
    }
    stats["sections_total"] = sum(indexed.chunk_count for indexed in indexed_files)
    if not enabled:
        return [], stats

    started = time.perf_counter()
    cache = _load_semantic_cache(persist_path)
    previous_entries = cache.get("entries", {})
    if not isinstance(previous_entries, dict):
        previous_entries = {}
    next_entries: dict[str, Any] = {}
    current_keys: set[str] = set()
    extracted_sections: list[dict[str, Any]] = []
    processed_sections = 0
    cache_flush_interval = max(1, int(os.getenv("SEMANTIC_CACHE_FLUSH_INTERVAL", "25")))
    progress_interval = max(1, int(os.getenv("SEMANTIC_PROGRESS_INTERVAL", "10")))
    last_progress_at = 0.0
    dirty_cache = False

    def persist_cache(force: bool = False) -> None:
        nonlocal dirty_cache
        if not dirty_cache:
            return
        _write_semantic_cache(
            persist_path,
            {
                "version": 1,
                "prompt_version": core.SEMANTIC_PROMPT_VERSION,
                "llm_model": llm_model,
                "llm_base_url": normalized_llm_base_url,
                "entries": next_entries,
            },
        )
        stats["cache_flushes"] += 1
        dirty_cache = False

    def emit_progress(force: bool = False) -> None:
        nonlocal last_progress_at
        if progress_callback is None or stats["sections_total"] <= 0:
            return
        now = time.perf_counter()
        if not force:
            if processed_sections == 0:
                return
            if processed_sections < stats["sections_total"]:
                if processed_sections % progress_interval != 0 and now - last_progress_at < 0.5:
                    return
        progress_callback(
            processed_sections,
            stats["sections_total"],
            _semantic_progress_message(processed_sections, stats["sections_total"], stats),
        )
        last_progress_at = now

    emit_progress(force=True)

    def record_result(
        *,
        section: dict[str, Any],
        cache_key: str,
        fingerprint: str,
        payload: dict[str, Any],
        usage: dict[str, int] | None = None,
        error: str | None = None,
        cached_entry: dict[str, Any] | None = None,
    ) -> None:
        nonlocal processed_sections, dirty_cache
        normalized_payload = payload if isinstance(payload, dict) else {"concepts": [], "decisions": []}
        if cached_entry is not None:
            stats["cached_sections"] += 1
            next_entries[cache_key] = cached_entry
        elif error is None:
            usage = usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            stats["api_calls"] += 1
            stats["extracted_sections"] += 1
            stats["prompt_tokens"] += usage["prompt_tokens"]
            stats["completion_tokens"] += usage["completion_tokens"]
            stats["total_tokens"] += usage["total_tokens"]
            next_entries[cache_key] = {
                "status": "ok",
                "fingerprint": fingerprint,
                "prompt_version": core.SEMANTIC_PROMPT_VERSION,
                "llm_model": llm_model,
                "llm_base_url": normalized_llm_base_url,
                "payload": normalized_payload,
                "usage": usage,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            dirty_cache = True
        else:
            stats["failed_sections"] += 1
            next_entries[cache_key] = {
                "status": "error",
                "fingerprint": fingerprint,
                "prompt_version": core.SEMANTIC_PROMPT_VERSION,
                "llm_model": llm_model,
                "llm_base_url": normalized_llm_base_url,
                "payload": normalized_payload,
                "error": error,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            dirty_cache = True
        if normalized_payload.get("concepts") or normalized_payload.get("decisions"):
            extracted_sections.append(
                {
                    "source": str(section["source"]),
                    "chunk_index": int(section["chunk_index"]),
                    "label": str(section["label"]),
                    "line_start": section.get("line_start"),
                    "line_end": section.get("line_end"),
                    "payload": normalized_payload,
                }
            )
        processed_sections += 1
        if dirty_cache and processed_sections % cache_flush_interval == 0:
            persist_cache()
        emit_progress()

    semantic_concurrency = max(1, int(os.getenv("SEMANTIC_CONCURRENCY", "1")))
    submitted_misses = 0
    worker_state = local()
    pending: dict[Any, dict[str, Any]] = {}

    def extract_with_client(section: dict[str, Any], llm: Any) -> dict[str, Any]:
        try:
            response = llm.invoke(_semantic_extraction_prompt(section))
            return {
                "section": section,
                "payload": _parse_semantic_payload(core._chunk_to_text(response)),
                "usage": _response_token_usage(response),
                "error": None,
            }
        except Exception as exc:
            return {
                "section": section,
                "payload": {"concepts": [], "decisions": []},
                "usage": None,
                "error": str(exc),
            }

    def worker_llm() -> Any:
        client = getattr(worker_state, "client", None)
        if client is None:
            client = core.make_llm(
                api_key=llm_api_key,
                model=llm_model,
                base_url=llm_base_url,
                temperature=0.0,
            )
            worker_state.client = client
        return client

    def extract_with_worker(section: dict[str, Any]) -> dict[str, Any]:
        return extract_with_client(section, worker_llm())

    def drain_completed(executor_done: set[Any]) -> None:
        for future in executor_done:
            meta = pending.pop(future)
            result = future.result()
            record_result(
                section=result["section"],
                cache_key=str(meta["cache_key"]),
                fingerprint=str(meta["fingerprint"]),
                payload=result["payload"],
                usage=result.get("usage"),
                error=result.get("error"),
            )

    executor: ThreadPoolExecutor | None = None
    if semantic_concurrency > 1:
        executor = ThreadPoolExecutor(max_workers=semantic_concurrency)

    try:
        for section in _iter_semantic_sections(indexed_files):
            cache_key = _semantic_cache_key(section)
            current_keys.add(cache_key)
            fingerprint = _semantic_section_fingerprint(
                source=str(section["source"]),
                chunk_index=int(section["chunk_index"]),
                text=str(section["text"]),
                llm_model=llm_model,
                llm_base_url=normalized_llm_base_url,
            )
            cached_entry = previous_entries.get(cache_key)
            if (
                isinstance(cached_entry, dict)
                and cached_entry.get("fingerprint") == fingerprint
                and cached_entry.get("prompt_version") == core.SEMANTIC_PROMPT_VERSION
                and cached_entry.get("llm_model") == llm_model
                and str(cached_entry.get("llm_base_url", "") or "") == normalized_llm_base_url
                and cached_entry.get("status") == "ok"
            ):
                payload = cached_entry.get("payload", {})
                if not isinstance(payload, dict):
                    payload = {"concepts": [], "decisions": []}
                    cached_entry = None
                if cached_entry is not None:
                    record_result(
                        section=section,
                        cache_key=cache_key,
                        fingerprint=fingerprint,
                        payload=payload,
                        cached_entry=cached_entry,
                    )
                    continue

            if executor is None:
                llm = worker_llm()
                result = extract_with_client(section, llm)
                record_result(
                    section=result["section"],
                    cache_key=cache_key,
                    fingerprint=fingerprint,
                    payload=result["payload"],
                    usage=result.get("usage"),
                    error=result.get("error"),
                )
                continue

            future = executor.submit(extract_with_worker, section)
            pending[future] = {
                "cache_key": cache_key,
                "fingerprint": fingerprint,
            }
            submitted_misses += 1
            if len(pending) >= semantic_concurrency:
                done, _pending = wait(set(pending), return_when=FIRST_COMPLETED)
                drain_completed(done)

        if executor is not None and pending:
            while pending:
                done, _pending = wait(set(pending), return_when=FIRST_COMPLETED)
                drain_completed(done)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if any(key not in current_keys for key in previous_entries):
        dirty_cache = True

    if submitted_misses > 0:
        stats["concurrency"] = min(semantic_concurrency, submitted_misses)
    else:
        stats["concurrency"] = 1

    stats["duration_seconds"] = round(time.perf_counter() - started, 3)
    persist_cache(force=True)
    emit_progress(force=True)
    return extracted_sections, stats


def _build_entity_graph(
    indexed_files: list[IndexedFile],
    document_graph: dict[str, Any],
    semantic_sections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    core = _core()
    nodes: list[dict[str, Any]] = []
    node_seen: set[str] = set()
    nodes_by_id: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[Any, ...], dict[str, Any]] = {}

    normalized_files: list[tuple[str, IndexedFile]] = []
    source_lookup: set[str] = set()
    basename_lookup: dict[str, list[str]] = {}
    for indexed in indexed_files:
        source = core._normalize_source_path(indexed.rel_path)
        if not source:
            continue
        normalized_files.append((source, indexed))
        source_lookup.add(source)
        basename_lookup.setdefault(Path(source).name, []).append(source)

    section_records_by_source: dict[str, list[dict[str, Any]]] = {}
    section_records_by_id: dict[str, dict[str, Any]] = {}
    symbol_records_by_source: dict[str, list[dict[str, Any]]] = {}
    all_symbol_entries: list[dict[str, Any]] = []
    pending_rationale_edges: list[dict[str, Any]] = []

    for source, indexed in sorted(normalized_files, key=lambda item: item[0]):
        total_lines = indexed.line_count
        kb = core._extract_kb(source) or indexed.kb
        core._add_entity_node(
            nodes,
            node_seen,
            {
                "id": core._entity_file_node_id(source),
                "type": "file",
                "name": Path(source).name,
                "source": source,
                "path": source,
                "kb": kb,
                "line_start": 1,
                "line_end": total_lines,
                "confidence": "EXTRACTED",
            },
        )
        nodes_by_id[core._entity_file_node_id(source)] = nodes[-1]

        section_records: list[dict[str, Any]] = []
        for chunk_index, chunk in enumerate(indexed.iter_chunks()):
            label = chunk.label or f"chunk {chunk_index + 1}"
            section_id = core._entity_section_node_id(source, chunk_index)
            section_line_start = chunk.line_start or 1
            section_line_end = chunk.line_end or section_line_start
            section_record = {
                "id": section_id,
                "type": "section",
                "name": label,
                "source": source,
                "file": source,
                "kb": kb,
                "chunk_index": chunk_index,
                "line_start": section_line_start,
                "line_end": section_line_end,
                "confidence": "EXTRACTED",
            }
            core._add_entity_node(nodes, node_seen, section_record)
            nodes_by_id[section_id] = nodes[-1]
            core._add_entity_edge(
                edges,
                core._entity_file_node_id(source),
                section_id,
                "contains",
                evidence_source=source,
                reason=label,
                line_start=section_line_start,
                line_end=section_line_end,
            )
            enriched_section_record = {**section_record, "chunk": chunk}
            section_records.append(enriched_section_record)
            section_records_by_id[section_id] = enriched_section_record
        section_records_by_source[source] = section_records

        symbol_records: list[dict[str, Any]] = []
        for record in indexed.symbols:
            symbol_source = core._normalize_source_path(record.get("source", source)) or source
            symbol_line_start = int(record.get("line_start") or 1)
            symbol_line_end = int(record.get("line_end") or symbol_line_start)
            symbol_record = {
                "id": core._entity_symbol_node_id(record),
                "type": "symbol",
                "name": str(record.get("name") or ""),
                "qualified_name": str(record.get("qualified_name") or record.get("name") or ""),
                "symbol_kind": str(record.get("kind") or "symbol"),
                "signature": str(record.get("signature") or ""),
                "source": symbol_source,
                "file": symbol_source,
                "kb": core._extract_kb(symbol_source) or kb,
                "line_start": symbol_line_start,
                "line_end": symbol_line_end,
                "confidence": "EXTRACTED",
            }
            core._add_entity_node(nodes, node_seen, symbol_record)
            nodes_by_id[str(symbol_record["id"])] = nodes[-1]
            core._add_entity_edge(
                edges,
                core._entity_file_node_id(source),
                str(symbol_record["id"]),
                "defines",
                evidence_source=source,
                reason=str(symbol_record["qualified_name"]),
                line_start=symbol_line_start,
                line_end=symbol_line_end,
            )
            symbol_records.append(symbol_record)
            all_symbol_entries.append(symbol_record)
        symbol_records_by_source[source] = symbol_records

    for source, section_records in section_records_by_source.items():
        symbol_records = symbol_records_by_source.get(source, [])
        for section_record in section_records:
            section_start = int(section_record["line_start"])
            section_end = int(section_record["line_end"])
            for symbol_record in symbol_records:
                if symbol_record.get("symbol_kind") == "module":
                    continue
                symbol_start = int(symbol_record["line_start"])
                if section_start <= symbol_start <= section_end:
                    core._add_entity_edge(
                        edges,
                        str(section_record["id"]),
                        str(symbol_record["id"]),
                        "contains",
                        evidence_source=source,
                        reason=str(section_record["name"]),
                        line_start=section_start,
                        line_end=section_end,
                    )

    symbol_lookup = _build_symbol_reference_lookup(all_symbol_entries)
    module_lookup = _build_python_module_lookup(indexed_files)

    for source, section_records in section_records_by_source.items():
        for section_record in section_records:
            chunk = section_record["chunk"]
            section_id = str(section_record["id"])
            line_start = int(section_record["line_start"])
            line_end = int(section_record["line_end"])
            for kind, reference in core._iter_local_path_references(chunk.text):
                target = core._resolve_document_reference(
                    source,
                    reference,
                    source_lookup,
                    basename_lookup,
                )
                if target is None:
                    continue
                core._add_entity_edge(
                    edges,
                    section_id,
                    core._entity_file_node_id(target),
                    kind,
                    evidence_source=source,
                    reason=reference,
                    line_start=line_start,
                    line_end=line_end,
                )

            for token in _extract_section_reference_tokens(chunk.text):
                target_symbol_id = _resolve_symbol_reference(source, token, symbol_lookup)
                if target_symbol_id is None:
                    continue
                core._add_entity_edge(
                    edges,
                    section_id,
                    target_symbol_id,
                    "references",
                    evidence_source=source,
                    reason=token,
                    line_start=line_start,
                    line_end=line_end,
                )

    for symbol_record in all_symbol_entries:
        if symbol_record.get("symbol_kind") != "import":
            continue
        target = _resolve_import_reference(
            str(symbol_record["source"]),
            str(symbol_record.get("qualified_name") or ""),
            module_lookup,
        )
        if target is None:
            continue
        core._add_entity_edge(
            edges,
            str(symbol_record["id"]),
            core._entity_file_node_id(target),
            "imports",
            evidence_source=str(symbol_record["source"]),
            reason=str(symbol_record.get("qualified_name") or ""),
            line_start=int(symbol_record["line_start"]),
            line_end=int(symbol_record["line_end"]),
        )

    concept_alias_lookup: dict[str, list[str]] = {}
    decision_alias_lookup: dict[str, list[str]] = {}
    semantic_sections = semantic_sections or []
    for semantic_section in semantic_sections:
        source = core._normalize_source_path(semantic_section.get("source", ""))
        chunk_index = int(semantic_section.get("chunk_index") or 0)
        section_id = core._entity_section_node_id(source, chunk_index)
        section_record = section_records_by_id.get(section_id)
        if not source or section_record is None:
            continue
        file_node_id = core._entity_file_node_id(source)
        line_start = int(section_record.get("line_start") or 1)
        line_end = int(section_record.get("line_end") or line_start)
        payload = semantic_section.get("payload")
        if not isinstance(payload, dict):
            continue

        for concept in payload.get("concepts", []):
            if not isinstance(concept, dict):
                continue
            concept_id = _upsert_semantic_node(
                nodes_by_id=nodes_by_id,
                nodes=nodes,
                node_seen=node_seen,
                alias_lookup=concept_alias_lookup,
                node_type="concept",
                item=concept,
            )
            reason = str(concept.get("name") or "")
            core._add_entity_edge(
                edges,
                section_id,
                concept_id,
                "semantically_related",
                evidence_source=source,
                reason=reason,
                line_start=line_start,
                line_end=line_end,
            )
            core._add_entity_edge(
                edges,
                file_node_id,
                concept_id,
                "semantically_related",
                evidence_source=source,
                reason=reason,
                line_start=line_start,
                line_end=line_end,
            )
            core._add_entity_edge(
                edges,
                concept_id,
                file_node_id,
                "semantically_related",
                evidence_source=source,
                reason=reason,
                line_start=line_start,
                line_end=line_end,
            )

        for decision in payload.get("decisions", []):
            if not isinstance(decision, dict):
                continue
            decision_id = _upsert_semantic_node(
                nodes_by_id=nodes_by_id,
                nodes=nodes,
                node_seen=node_seen,
                alias_lookup=decision_alias_lookup,
                node_type="decision",
                item=decision,
            )
            reason = str(decision.get("name") or "")
            core._add_entity_edge(
                edges,
                section_id,
                decision_id,
                "semantically_related",
                evidence_source=source,
                reason=reason,
                line_start=line_start,
                line_end=line_end,
            )
            core._add_entity_edge(
                edges,
                file_node_id,
                decision_id,
                "semantically_related",
                evidence_source=source,
                reason=reason,
                line_start=line_start,
                line_end=line_end,
            )
            core._add_entity_edge(
                edges,
                decision_id,
                file_node_id,
                "semantically_related",
                evidence_source=source,
                reason=reason,
                line_start=line_start,
                line_end=line_end,
            )
            rationale_values = decision.get("rationale")
            if isinstance(rationale_values, list):
                pending_rationale_edges.append(
                    {
                        "decision_id": decision_id,
                        "rationale": list(rationale_values),
                        "source": source,
                        "line_start": line_start,
                        "line_end": line_end,
                    }
                )

    for pending_edge in pending_rationale_edges:
        decision_id = str(pending_edge.get("decision_id", "") or "")
        source = str(pending_edge.get("source", "") or "")
        line_start = int(pending_edge.get("line_start") or 1)
        line_end = int(pending_edge.get("line_end") or line_start)
        rationale_values = pending_edge.get("rationale")
        if not decision_id or not isinstance(rationale_values, list):
            continue
        for rationale in rationale_values:
            rationale_key = str(rationale or "").strip().lower()
            if not rationale_key:
                continue
            concept_ids = concept_alias_lookup.get(rationale_key, [])
            if not concept_ids:
                continue
            for concept_id in sorted(set(concept_ids)):
                core._add_entity_edge(
                    edges,
                    concept_id,
                    decision_id,
                    "rationale_for",
                    evidence_source=source,
                    reason=str(rationale),
                    line_start=line_start,
                    line_end=line_end,
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
        "version": 1,
        "node_count": len(sorted_nodes),
        "edge_count": len(sorted_edges),
        "nodes": sorted_nodes,
        "edges": sorted_edges,
    }
