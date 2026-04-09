from __future__ import annotations

import ast
import configparser
import fnmatch
import json
import os
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    yaml = None
    _YAML_AVAILABLE = False

try:
    from markitdown import MarkItDown

    _MARKITDOWN_AVAILABLE = True
except ImportError:
    MarkItDown = None
    _MARKITDOWN_AVAILABLE = False

if TYPE_CHECKING:
    from ragbot import IndexedFile, SearchBundle


def _core():
    import ragbot as core

    return core


def _safe_signature_from_args(node: ast.AST) -> str:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    args = []
    for arg in node.args.posonlyargs:
        args.append(arg.arg)
    for arg in node.args.args:
        args.append(arg.arg)
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    for arg in node.args.kwonlyargs:
        args.append(arg.arg)
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    return ", ".join(args)


def _should_ignore_relative_path(rel_path: Path, extra_patterns: list[str]) -> bool:
    core = _core()
    parts = set(rel_path.parts)
    if parts & core.DEFAULT_IGNORED_DIRS:
        return True
    posix_path = rel_path.as_posix()
    for pattern in extra_patterns:
        if fnmatch.fnmatch(posix_path, pattern) or fnmatch.fnmatch(rel_path.name, pattern):
            return True
    return False


def _iter_supported_files(source_dir: Path) -> list[Path]:
    core = _core()
    files: list[Path] = []
    extra_patterns = core._load_extra_exclude_globs()

    for dirpath, dirnames, filenames in os.walk(source_dir):
        current_dir = Path(dirpath)
        rel_dir = current_dir.relative_to(source_dir)
        kept_dirs: list[str] = []
        for dirname in dirnames:
            rel_child = rel_dir / dirname
            if _should_ignore_relative_path(rel_child, extra_patterns):
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in filenames:
            path = current_dir / filename
            rel_path = path.relative_to(source_dir)
            if _should_ignore_relative_path(rel_path, extra_patterns):
                continue
            if path.suffix.lower() in core.SUPPORTED_TEXT_SUFFIXES:
                files.append(path)

    return sorted(files)


def _normalize_structured_text(text: str, suffix: str) -> str:
    core = _core()
    try:
        if suffix == ".json":
            parsed = json.loads(text)
            return "\n".join(core._flatten_mapping(parsed))
        if suffix in {".yaml", ".yml"} and _YAML_AVAILABLE:
            parsed = yaml.safe_load(text)
            return "\n".join(core._flatten_mapping(parsed))
        if suffix == ".toml":
            parsed = tomllib.loads(text)
            return "\n".join(core._flatten_mapping(parsed))
        if suffix == ".ini":
            parser = configparser.ConfigParser()
            parser.optionxform = str
            parser.read_string(text)
            data: dict[str, Any] = {}
            for section in parser.sections():
                data[section] = dict(parser.items(section))
            return "\n".join(core._flatten_mapping(data))
    except Exception:
        pass
    return core._normalize_text(text)


def _chunk_structured_text(normalized_text: str) -> list[Any]:
    core = _core()
    return core._split_lines_into_chunks(normalized_text.splitlines())


def _convert_binary_to_markdown(file_path: Path) -> str:
    if not _MARKITDOWN_AVAILABLE:
        raise ImportError(
            f"处理 {file_path.suffix} 文件需要 markitdown 库，"
            "请运行: pip install 'markitdown[pdf,docx,xlsx]'"
        )
    md = MarkItDown()
    result = md.convert(str(file_path))
    return result.text_content


def _chunk_markdown_by_heading(text: str) -> list[Any]:
    core = _core()
    lines = text.splitlines()
    if not lines:
        return []

    heading_indexes = [
        idx for idx, line in enumerate(lines) if core.re.match(r"^\s{0,3}#{1,6}\s+", line)
    ]
    if not heading_indexes:
        return core._split_lines_into_chunks(lines)

    chunks: list[Any] = []
    starts = heading_indexes + [len(lines)]
    for current, next_index in zip(starts, starts[1:]):
        section_lines = lines[current:next_index]
        section_text = "\n".join(section_lines).strip()
        if not section_text:
            continue
        heading_line = section_lines[0].strip()
        label = heading_line.lstrip("#").strip() or f"section {len(chunks) + 1}"
        chunks.append(
            core.ChunkSpec(
                text=section_text,
                line_start=current + 1,
                line_end=next_index,
                label=label,
            )
        )

    preface_end = heading_indexes[0]
    if preface_end > 0:
        preface_lines = lines[:preface_end]
        preface_text = "\n".join(preface_lines).strip()
        if preface_text:
            chunks.insert(
                0,
                core.ChunkSpec(
                    text=preface_text,
                    line_start=1,
                    line_end=preface_end,
                    label="preface",
                ),
            )

    return core._ensure_chunk_size(chunks)


def _chunk_python_code(text: str) -> list[Any]:
    core = _core()
    normalized = core._normalize_text(text)
    try:
        tree = ast.parse(normalized)
    except SyntaxError:
        return core._split_lines_into_chunks(normalized.splitlines())

    lines = normalized.splitlines()
    body_nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    if not body_nodes:
        return core._split_lines_into_chunks(lines)

    chunks: list[Any] = []
    preface_end = body_nodes[0].lineno - 1
    if preface_end > 0:
        chunks.extend(
            core._split_lines_into_chunks(lines[:preface_end], start_line=1, label="module prelude")
        )

    for node in body_nodes:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", start)
        if start is None:
            continue
        node_lines = lines[start - 1 : end]
        label = getattr(node, "name", f"block {len(chunks) + 1}")
        if isinstance(node, ast.ClassDef):
            label = f"class {label}"
        else:
            label = f"def {label}"
        chunks.append(
            core.ChunkSpec(
                text="\n".join(node_lines).strip(),
                line_start=start,
                line_end=end,
                label=label,
            )
        )

    return core._ensure_chunk_size(chunks)


def _extract_python_symbols(text: str, rel_path: str) -> list[dict[str, Any]]:
    core = _core()
    normalized = core._normalize_text(text)
    try:
        tree = ast.parse(normalized)
    except SyntaxError:
        return []

    records: list[dict[str, Any]] = []
    total_lines = max(1, len(normalized.splitlines()))
    records.append(
        {
            "kind": "module",
            "name": Path(rel_path).stem,
            "qualified_name": Path(rel_path).stem,
            "source": rel_path,
            "line_start": 1,
            "line_end": total_lines,
            "signature": f"module {rel_path}",
        }
    )

    class Collector(ast.NodeVisitor):
        def __init__(self):
            self.stack: list[str] = []

        def _qualified_name(self, name: str) -> str:
            return ".".join([*self.stack, name]) if self.stack else name

        def visit_ClassDef(self, node: ast.ClassDef) -> Any:
            qualified = self._qualified_name(node.name)
            records.append(
                {
                    "kind": "class",
                    "name": node.name,
                    "qualified_name": qualified,
                    "source": rel_path,
                    "line_start": node.lineno,
                    "line_end": getattr(node, "end_lineno", node.lineno),
                    "signature": f"class {node.name}",
                }
            )
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            qualified = self._qualified_name(node.name)
            signature = _safe_signature_from_args(node)
            records.append(
                {
                    "kind": "function",
                    "name": node.name,
                    "qualified_name": qualified,
                    "source": rel_path,
                    "line_start": node.lineno,
                    "line_end": getattr(node, "end_lineno", node.lineno),
                    "signature": f"def {qualified}({signature})",
                }
            )
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
            qualified = self._qualified_name(node.name)
            signature = _safe_signature_from_args(node)
            records.append(
                {
                    "kind": "function",
                    "name": node.name,
                    "qualified_name": qualified,
                    "source": rel_path,
                    "line_start": node.lineno,
                    "line_end": getattr(node, "end_lineno", node.lineno),
                    "signature": f"async def {qualified}({signature})",
                }
            )
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Import(self, node: ast.Import) -> Any:
            for alias in node.names:
                records.append(
                    {
                        "kind": "import",
                        "name": alias.asname or alias.name,
                        "qualified_name": alias.name,
                        "source": rel_path,
                        "line_start": node.lineno,
                        "line_end": getattr(node, "end_lineno", node.lineno),
                        "signature": f"import {alias.name}",
                    }
                )

        def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
            module = node.module or ""
            for alias in node.names:
                name = alias.asname or alias.name
                qualified = f"{module}.{alias.name}".strip(".")
                records.append(
                    {
                        "kind": "import",
                        "name": name,
                        "qualified_name": qualified,
                        "source": rel_path,
                        "line_start": node.lineno,
                        "line_end": getattr(node, "end_lineno", node.lineno),
                        "signature": f"from {module} import {alias.name}",
                    }
                )

    Collector().visit(tree)
    return records


def _process_source_file(root: Path, file_path: Path) -> IndexedFile | None:
    core = _core()
    rel_path = core._normalize_source_path(file_path.relative_to(root).as_posix())
    kb = core._extract_kb(rel_path)
    suffix = file_path.suffix.lower()

    try:
        if suffix in core._BINARY_SUFFIXES:
            raw_text = core._convert_binary_to_markdown(file_path)
        else:
            raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    normalized_raw = core._normalize_text(raw_text)
    if not normalized_raw:
        return None

    symbols: list[dict[str, Any]] = []
    if suffix in core.MARKDOWN_SUFFIXES or suffix in core._BINARY_SUFFIXES:
        messages = core._parse_md_messages_text(normalized_raw) if suffix in core.MARKDOWN_SUFFIXES else []
        if messages:
            chunk_specs: list[Any] = []
            for idx, chunk in enumerate(core._chunk_by_time_window(messages), start=1):
                chunk_lines = []
                for message in chunk:
                    ts = message["timestamp"].strftime("%H:%M") if message.get("timestamp") else "?"
                    chunk_lines.append(f"[{ts}] {message['sender']}: {message['content']}")
                text = "\n".join(chunk_lines).strip()
                if not text:
                    continue
                timestamps = [item["timestamp"] for item in chunk if item.get("timestamp")]
                start_t = timestamps[0].strftime("%Y-%m-%d %H:%M") if timestamps else ""
                end_t = timestamps[-1].strftime("%H:%M") if timestamps else ""
                label = f"{start_t} ~ {end_t}" if start_t else f"window {idx}"
                line_start = chunk[0].get("line_start")
                line_end = chunk[-1].get("line_end")
                chunk_specs.append(
                    core.ChunkSpec(
                        text=text,
                        line_start=line_start,
                        line_end=line_end,
                        label=label,
                    )
                )
            chunks = core._ensure_chunk_size(chunk_specs)
        else:
            chunks = core._chunk_markdown_by_heading(normalized_raw)
        normalized_text = normalized_raw
    elif suffix in core.STRUCTURED_SUFFIXES:
        normalized_text = core._normalize_structured_text(normalized_raw, suffix)
        chunks = core._chunk_structured_text(normalized_text)
    elif suffix in core.PYTHON_SUFFIXES:
        normalized_text = normalized_raw
        chunks = core._chunk_python_code(normalized_raw)
        symbols = core._extract_python_symbols(normalized_raw, rel_path)
    else:
        normalized_text = normalized_raw
        chunks = core._split_lines_into_chunks(normalized_text.splitlines())

    if not chunks:
        chunks = core._split_lines_into_chunks(normalized_text.splitlines())
    if not chunks:
        return None

    return core.IndexedFile(
        rel_path=rel_path,
        suffix=suffix,
        kb=kb,
        file_path=file_path,
        normalized_text=normalized_text,
        chunks=chunks,
        symbols=symbols,
    )


def _build_documents(source_dir: str) -> tuple[list[Document], list[IndexedFile]]:
    core = _core()
    root = Path(source_dir)
    files = core._iter_supported_files(root)
    documents: list[Document] = []
    indexed_files: list[IndexedFile] = []

    for file_path in files:
        processed = core._process_source_file(root, file_path)
        if processed is None:
            continue

        indexed_files.append(processed)
        for idx, chunk in enumerate(processed.chunks):
            documents.append(
                Document(
                    page_content=chunk.text,
                    metadata={
                        "source": processed.rel_path,
                        "kb": processed.kb,
                        "chunk_index": idx,
                        "time_range": core._chunk_location_label(chunk, f"chunk {idx + 1}"),
                        "line_start": chunk.line_start,
                        "line_end": chunk.line_end,
                    },
                )
            )

    return documents, indexed_files


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
        for kind, reference in core._iter_local_path_references(indexed.normalized_text):
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


def _write_index_artifacts(
    persist_path: Path,
    indexed_files: list[IndexedFile],
    embed_model: str,
    source_dir: str,
    total_chunks: int,
    llm_api_key: str = "",
    llm_model: str = "",
    llm_base_url: str = "",
) -> dict[str, Any]:
    core = _core()
    normalized_dir = persist_path / core.NORMALIZED_TEXT_DIRNAME
    if normalized_dir.exists():
        core.shutil.rmtree(normalized_dir)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    symbol_records: list[dict[str, Any]] = []
    manifest = {
        "build_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "embed_model": embed_model,
        "total_chunks": total_chunks,
        "kb_enabled": True,
        "source_dir": str(Path(source_dir).expanduser()),
        "normalized_text_dir": core.NORMALIZED_TEXT_DIRNAME,
        "symbol_index_file": core.SYMBOL_INDEX_FILENAME,
        "document_graph_file": core.DOCUMENT_GRAPH_FILENAME,
        "entity_graph_file": core.ENTITY_GRAPH_FILENAME,
        "semantic_extract_cache_file": core.SEMANTIC_EXTRACT_CACHE_FILENAME,
        "community_index_file": core.COMMUNITY_INDEX_FILENAME,
        "graph_report_file": f"{core.REPORTS_DIRNAME}/{core.GRAPH_REPORT_FILENAME}",
        "lint_report_file": core.LINT_REPORT_FILENAME,
        "search_mode_default": os.getenv("SEARCH_MODE", core.DEFAULT_SEARCH_MODE).strip() or core.DEFAULT_SEARCH_MODE,
        "semantic_graph_stats": {},
        "files": [],
    }

    for indexed in indexed_files:
        normalized_rel = f"{indexed.rel_path}.txt"
        cache_path = normalized_dir / normalized_rel
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(indexed.normalized_text, encoding="utf-8")

        stat = indexed.file_path.stat()
        manifest["files"].append(
            {
                "name": indexed.rel_path,
                "kb": indexed.kb,
                "suffix": indexed.suffix,
                "size_kb": round(stat.st_size / 1024, 1),
                "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "chunks": len(indexed.chunks),
                "normalized_text": normalized_rel,
            }
        )
        symbol_records.extend(indexed.symbols)

    symbol_index_path = persist_path / core.SYMBOL_INDEX_FILENAME
    if symbol_records:
        symbol_index_path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in symbol_records),
            encoding="utf-8",
        )
    elif symbol_index_path.exists():
        symbol_index_path.unlink()

    document_graph = core._build_document_graph(indexed_files)
    document_graph_path = persist_path / core.DOCUMENT_GRAPH_FILENAME
    document_graph_path.write_text(
        json.dumps(document_graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    semantic_sections, semantic_stats = core._extract_semantic_sections(
        indexed_files=indexed_files,
        persist_path=persist_path,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
    )
    manifest["semantic_graph_stats"] = semantic_stats

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

    file_sources = [core._normalize_source_path(indexed.rel_path) for indexed in indexed_files]
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
    documents, indexed_files = core._build_documents(md_dir)
    total = len(documents)

    if total == 0:
        raise ValueError(
            f"在 {md_dir} 中未找到可索引文本。支持后缀: {', '.join(sorted(core.SUPPORTED_TEXT_SUFFIXES))}"
        )

    _cb(0, total + 4, f"解析完毕，共 {len(indexed_files)} 个文件，{total} 个分片。")
    embeddings = core.make_embeddings(api_key=embed_api_key, base_url=embed_base_url, model=embed_model)

    batch_size = max(1, int(os.getenv("EMBED_BATCH_SIZE", str(core.DEFAULT_EMBED_BATCH_SIZE))))
    # 默认不做固定节流；仅在真正触发限流时才指数退避，避免大语料重建被 sleep 吞掉数小时。
    batch_sleep_seconds = max(
        0.0,
        float(
            os.getenv(
                "EMBED_BATCH_SLEEP_SECONDS",
                str(core.DEFAULT_EMBED_BATCH_SLEEP_SECONDS),
            )
        ),
    )
    max_retries = max(0, int(os.getenv("EMBED_MAX_RETRIES", "8")))
    retry_base_seconds = max(0.5, float(os.getenv("EMBED_RETRY_BASE_SECONDS", "5")))

    vectorstore: FAISS | None = None
    total_steps = total + 4
    for i in range(0, total, batch_size):
        batch = documents[i : i + batch_size]
        for attempt in range(max_retries + 1):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.add_documents(batch)
                break
            except Exception as exc:
                if not core._is_rate_limit_error(exc) or attempt >= max_retries:
                    raise
                delay = retry_base_seconds * (2**attempt)
                _cb(
                    i,
                    total_steps,
                    f"触发 embedding 限流，{delay:.1f}s 后重试第 {attempt + 1}/{max_retries} 次...",
                )
                core.time.sleep(delay)
        done = min(i + batch_size, total)
        _cb(done, total_steps, f"向量化中 {done}/{total}...")
        if batch_sleep_seconds > 0 and done < total:
            core.time.sleep(batch_sleep_seconds)

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(persist_path))  # type: ignore[union-attr]
    _cb(total + 1, total_steps, "正在写入检索辅助索引...")

    manifest = core._write_index_artifacts(
        persist_path=persist_path,
        indexed_files=indexed_files,
        embed_model=embed_model,
        source_dir=md_dir,
        total_chunks=total,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
    )
    _cb(total + 2, total_steps, "正在生成离线 wiki 导航...")
    from wiki import generate_wiki

    generate_wiki(persist_path=persist_path, manifest=manifest)
    core._read_cached_text.cache_clear()
    _cb(total + 3, total_steps, "正在加载检索 bundle...")

    bundle = core.load_search_bundle(
        embed_api_key=embed_api_key,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        persist_dir=persist_dir,
    )
    if bundle is None:
        raise RuntimeError("索引文件写入成功，但 SearchBundle 回读失败。")

    _cb(total + 4, total_steps, f"✅ 索引构建完成，已保存到 {persist_dir}")
    bundle.manifest = manifest
    return bundle
