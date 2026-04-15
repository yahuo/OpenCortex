from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ragbot_sources import _build_documents

if TYPE_CHECKING:
    from ragbot import SearchBundle


def _core():
    import ragbot as core

    return core


_ORIGINAL_BUILD_DOCUMENTS = _build_documents


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
