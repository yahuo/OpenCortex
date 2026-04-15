from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import os
from queue import Empty, Queue
import shutil
import tempfile
from threading import local
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from ragbot_artifacts import (
    _source_snapshot_from_file_records,
    atomic_write_text,
    current_build_config_snapshot,
)
from ragbot_sources import _build_documents

if TYPE_CHECKING:
    from ragbot import SearchBundle


def _core():
    import ragbot as core

    return core
_ORIGINAL_BUILD_DOCUMENTS = _build_documents


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
        "search_mode_default": os.getenv("SEARCH_MODE", core.DEFAULT_SEARCH_MODE).strip() or core.DEFAULT_SEARCH_MODE,
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


def _iter_document_batches(documents: Iterable[Document], batch_size: int) -> Iterable[list[Document]]:
    batch: list[Document] = []
    for document in documents:
        batch.append(document)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _document_batch_ids(batch: list[Document]) -> list[str] | None:
    ids = [getattr(document, "id", None) for document in batch]
    if any(ids):
        return [str(doc_id or "") for doc_id in ids]
    return None


def _embed_texts_with_retries(
    *,
    texts: list[str],
    embeddings_factory: Callable[[], OpenAIEmbeddings],
    max_retries: int,
    retry_base_seconds: float,
    rate_limit_callback: Callable[[float, int, int], None] | None = None,
) -> list[list[float]]:
    core = _core()
    for attempt in range(max_retries + 1):
        try:
            return embeddings_factory().embed_documents(texts)
        except Exception as exc:
            if not core._is_rate_limit_error(exc) or attempt >= max_retries:
                raise
            delay = retry_base_seconds * (2**attempt)
            if rate_limit_callback is not None:
                rate_limit_callback(delay, attempt + 1, max_retries)
            core.time.sleep(delay)
    raise RuntimeError("embedding 批处理在重试后仍未成功。")


def _add_embedded_batch(
    *,
    batch: list[Document],
    embedded_texts: list[list[float]],
    embeddings: OpenAIEmbeddings,
    vectorstore: FAISS | None,
) -> FAISS:
    core = _core()
    texts = [document.page_content for document in batch]
    metadatas = [document.metadata for document in batch]
    ids = _document_batch_ids(batch)
    text_embeddings = zip(texts, embedded_texts)
    if vectorstore is None:
        if not hasattr(core.FAISS, "from_embeddings"):
            return core.FAISS.from_documents(batch, embeddings)
        return core.FAISS.from_embeddings(
            text_embeddings,
            embeddings,
            metadatas=metadatas,
            ids=ids,
        )
    if not hasattr(vectorstore, "add_embeddings"):
        vectorstore.add_documents(batch)
        return vectorstore
    vectorstore.add_embeddings(
        text_embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    return vectorstore


def _build_vectorstore_from_document_stream(
    *,
    indexed_files: list[IndexedFile],
    documents: Iterable[Document],
    total_chunks: int,
    md_dir: str,
    embed_api_key: str,
    embed_base_url: str,
    embed_model: str,
    persist_dir: str,
    staged_normalized_dir: Path | None,
    progress_callback: Callable[[int, int, str], None] | None,
    llm_api_key: str,
    llm_model: str,
    llm_base_url: str,
) -> SearchBundle:
    core = _core()

    def _cb(current: int, total: int, message: str) -> None:
        if progress_callback:
            progress_callback(current, total, message)

    if total_chunks == 0:
        raise ValueError(
            f"在 {md_dir} 中未找到可索引文本。支持后缀: {', '.join(sorted(core.SUPPORTED_TEXT_SUFFIXES))}"
        )

    _cb(0, total_chunks + 4, f"解析完毕，共 {len(indexed_files)} 个文件，{total_chunks} 个分片。")
    truncated_files = [indexed for indexed in indexed_files if indexed.truncated]
    if truncated_files:
        preview = "；".join(
            f"{indexed.rel_path}: {indexed.original_chunk_count} -> {indexed.chunk_count}"
            for indexed in truncated_files[:3]
        )
        _cb(
            0,
            total_chunks + 4,
            f"注意：{len(truncated_files)} 个文件因 MAX_CHUNKS_PER_FILE 被截断。{preview}",
        )
    embeddings = core.make_embeddings(api_key=embed_api_key, base_url=embed_base_url, model=embed_model)

    batch_size = max(1, int(os.getenv("EMBED_BATCH_SIZE", str(core.DEFAULT_EMBED_BATCH_SIZE))))
    batch_sleep_seconds = max(
        0.0,
        float(os.getenv("EMBED_BATCH_SLEEP_SECONDS", str(core.DEFAULT_EMBED_BATCH_SLEEP_SECONDS))),
    )
    embed_concurrency = max(1, int(os.getenv("EMBED_CONCURRENCY", "1")))
    max_retries = max(0, int(os.getenv("EMBED_MAX_RETRIES", "8")))
    retry_base_seconds = max(0.5, float(os.getenv("EMBED_RETRY_BASE_SECONDS", "5")))

    vectorstore: FAISS | None = None
    total_steps = total_chunks + 4
    processed_chunks = 0
    rate_limit_events: Queue[str] = Queue()

    def _emit_rate_limit_event(delay: float, attempt: int, retry_limit: int) -> None:
        message = f"触发 embedding 限流，{delay:.1f}s 后重试第 {attempt}/{retry_limit} 次..."
        if embed_concurrency <= 1:
            _cb(processed_chunks, total_steps, message)
            return
        rate_limit_events.put(message)

    def _drain_rate_limit_events() -> None:
        while True:
            try:
                message = rate_limit_events.get_nowait()
            except Empty:
                break
            _cb(processed_chunks, total_steps, message)

    if embed_concurrency <= 1:
        for batch in _iter_document_batches(documents, batch_size):
            embedded_texts = _embed_texts_with_retries(
                texts=[document.page_content for document in batch],
                embeddings_factory=lambda: embeddings,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                rate_limit_callback=_emit_rate_limit_event,
            )
            vectorstore = _add_embedded_batch(
                batch=batch,
                embedded_texts=embedded_texts,
                embeddings=embeddings,
                vectorstore=vectorstore,
            )
            processed_chunks += len(batch)
            _cb(processed_chunks, total_steps, f"向量化中 {processed_chunks}/{total_chunks}...")
            if batch_sleep_seconds > 0 and processed_chunks < total_chunks:
                core.time.sleep(batch_sleep_seconds)
    else:
        worker_state = local()
        if batch_sleep_seconds > 0:
            _cb(
                processed_chunks,
                total_steps,
                "EMBED_BATCH_SLEEP_SECONDS 仅在串行 embedding 下生效；并发模式下已忽略。",
            )

        def _worker_embeddings() -> OpenAIEmbeddings:
            client = getattr(worker_state, "client", None)
            if client is None:
                client = core.make_embeddings(
                    api_key=embed_api_key,
                    base_url=embed_base_url,
                    model=embed_model,
                )
                worker_state.client = client
            return client

        def _submit_embed_batch(
            executor: ThreadPoolExecutor,
            pending: dict[int, tuple[list[Document], Future[list[list[float]]]]],
            batch_iter: Iterable[list[Document]],
            seq: int,
        ) -> tuple[bool, int]:
            try:
                batch = next(batch_iter)
            except StopIteration:
                return False, seq
            pending[seq] = (
                batch,
                executor.submit(
                    _embed_texts_with_retries,
                    texts=[document.page_content for document in batch],
                    embeddings_factory=_worker_embeddings,
                    max_retries=max_retries,
                    retry_base_seconds=retry_base_seconds,
                    rate_limit_callback=_emit_rate_limit_event,
                ),
            )
            return True, seq + 1

        batch_iter = iter(_iter_document_batches(documents, batch_size))
        pending: dict[int, tuple[list[Document], Future[list[list[float]]]]] = {}
        next_seq = 0
        expected_seq = 0
        with ThreadPoolExecutor(max_workers=embed_concurrency) as executor:
            while len(pending) < embed_concurrency:
                submitted, next_seq = _submit_embed_batch(executor, pending, batch_iter, next_seq)
                if not submitted:
                    break

            while pending:
                batch, future = pending.pop(expected_seq)
                while True:
                    try:
                        embedded_texts = future.result(timeout=0.1)
                        break
                    except FutureTimeoutError:
                        _drain_rate_limit_events()
                _drain_rate_limit_events()
                vectorstore = _add_embedded_batch(
                    batch=batch,
                    embedded_texts=embedded_texts,
                    embeddings=embeddings,
                    vectorstore=vectorstore,
                )
                processed_chunks += len(batch)
                _cb(processed_chunks, total_steps, f"向量化中 {processed_chunks}/{total_chunks}...")
                expected_seq += 1

                while len(pending) < embed_concurrency:
                    submitted, next_seq = _submit_embed_batch(executor, pending, batch_iter, next_seq)
                    if not submitted:
                        break

    if vectorstore is None:
        raise RuntimeError("未生成任何向量分片。")

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(persist_path))
    _cb(total_chunks + 1, total_steps, "正在写入检索辅助索引...")

    manifest = core._write_index_artifacts(
        persist_path=persist_path,
        indexed_files=indexed_files,
        embed_model=embed_model,
        embed_base_url=embed_base_url,
        source_dir=md_dir,
        total_chunks=total_chunks,
        staged_normalized_dir=staged_normalized_dir,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
    )
    if manifest.get("build_flags", {}).get("skip_wiki"):
        core._clear_generated_wiki_artifacts(persist_path)
        _cb(total_chunks + 2, total_steps, "已按配置跳过离线 wiki 生成。")
    else:
        _cb(total_chunks + 2, total_steps, "正在生成离线 wiki 导航...")
        from wiki import generate_wiki

        generate_wiki(persist_path=persist_path, manifest=manifest)
    core._read_cached_text.cache_clear()
    _cb(total_chunks + 3, total_steps, "正在加载检索 bundle...")

    bundle = core.load_search_bundle(
        embed_api_key=embed_api_key,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        persist_dir=persist_dir,
    )
    if bundle is None:
        raise RuntimeError("索引文件写入成功，但 SearchBundle 回读失败。")

    _cb(total_chunks + 4, total_steps, f"✅ 索引构建完成，已保存到 {persist_dir}")
    bundle.manifest = manifest
    return bundle


def build_vectorstore(
    md_dir: str,
    embed_api_key: str,
    embed_base_url: str = "https://api.siliconflow.cn/v1",
    embed_model: str = "BAAI/bge-m3",
    persist_dir: str = str(Path.home() / "wechat_rag_db"),
    progress_callback: Callable[[int, int, str], None] | None = None,
    llm_api_key: str = "",
    llm_model: str = "",
    llm_base_url: str = "",
) -> SearchBundle:
    core = _core()

    def _cb(current: int, total: int, message: str) -> None:
        if progress_callback:
            progress_callback(current, total, message)

    _cb(0, 1, "正在扫描目录并解析文本文件...")
    if core._build_documents is not _ORIGINAL_BUILD_DOCUMENTS:
        documents, indexed_files = core._build_documents(md_dir)
        return core._build_vectorstore_from_document_stream(
            indexed_files=indexed_files,
            documents=documents,
            total_chunks=len(documents),
            md_dir=md_dir,
            embed_api_key=embed_api_key,
            embed_base_url=embed_base_url,
            embed_model=embed_model,
            persist_dir=persist_dir,
            staged_normalized_dir=None,
            progress_callback=progress_callback,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
        )

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".opencortex-build-", dir=str(persist_path)) as tmpdir:
        cache_root = Path(tmpdir) / core.NORMALIZED_TEXT_DIRNAME
        indexed_files, total_chunks = core._index_source_files(md_dir, cache_root)
        return core._build_vectorstore_from_document_stream(
            indexed_files=indexed_files,
            documents=core._iter_documents_for_indexed_files(indexed_files),
            total_chunks=total_chunks,
            md_dir=md_dir,
            embed_api_key=embed_api_key,
            embed_base_url=embed_base_url,
            embed_model=embed_model,
            persist_dir=persist_dir,
            staged_normalized_dir=cache_root,
            progress_callback=progress_callback,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
        )
