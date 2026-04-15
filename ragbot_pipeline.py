from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import os
from queue import Empty, Queue
from threading import local
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

if TYPE_CHECKING:
    from ragbot import IndexedFile, SearchBundle


def _core():
    import ragbot as core

    return core


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
