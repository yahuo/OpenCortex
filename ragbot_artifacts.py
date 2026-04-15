from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

BASE_RUNTIME_INDEX_ARTIFACTS = ("index.faiss", "index.pkl", "index_manifest.json")


def _core():
    import ragbot as core

    return core


def load_index_manifest(persist_dir: str | Path) -> dict[str, Any] | None:
    manifest_path = Path(persist_dir) / "index_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def runtime_index_artifacts(
    persist_dir: str | Path,
    manifest: dict[str, Any] | None = None,
) -> list[Path]:
    core = _core()
    persist_path = Path(persist_dir)
    artifacts = [persist_path / name for name in BASE_RUNTIME_INDEX_ARTIFACTS]
    resolved_manifest = manifest if manifest is not None else load_index_manifest(persist_path)
    fulltext_relpath = core.FULLTEXT_INDEX_FILENAME
    if isinstance(resolved_manifest, dict):
        fulltext_relpath = str(
            resolved_manifest.get("fulltext_index_file", core.FULLTEXT_INDEX_FILENAME)
            or core.FULLTEXT_INDEX_FILENAME
        )
    artifacts.append(persist_path / Path(fulltext_relpath))
    return artifacts


def fulltext_index_path(
    persist_dir: str | Path,
    manifest: dict[str, Any] | None = None,
) -> Path:
    return runtime_index_artifacts(persist_dir, manifest=manifest)[-1]


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


def current_build_snapshot(
    source_dir: str,
    *,
    embed_base_url: str,
    embed_model: str,
    llm_api_key: str = "",
    llm_model: str = "",
    llm_base_url: str = "",
) -> dict[str, Any]:
    snapshot = current_build_config_snapshot(
        source_dir,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
    )
    source_snapshot = current_source_snapshot(source_dir)
    snapshot.update(
        {
            "file_count": int(source_snapshot.get("file_count") or 0),
            "source_digest": str(source_snapshot.get("source_digest", "") or ""),
        }
    )
    return snapshot


def current_build_config_snapshot(
    source_dir: str,
    *,
    embed_base_url: str,
    embed_model: str,
    llm_api_key: str = "",
    llm_model: str = "",
    llm_base_url: str = "",
) -> dict[str, Any]:
    core = _core()
    root = Path(source_dir).expanduser()
    semantic_enabled, semantic_reason = core._semantic_graph_enabled(
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
    )
    return {
        "version": 1,
        "source_dir": str(root),
        "embed_model": embed_model.strip(),
        "embed_base_url": embed_base_url.strip().rstrip("/"),
        "llm_model": llm_model.strip(),
        "llm_base_url": llm_base_url.strip().rstrip("/"),
        "chunk_size": core._configured_chunk_size(),
        "chunk_overlap_lines": core._configured_chunk_overlap_lines(),
        "max_chunks_per_file": core._configured_max_chunks_per_file(),
        "exclude_globs": sorted(core._load_extra_exclude_globs()),
        "skip_graph": core._env_flag("SKIP_GRAPH"),
        "skip_semantic": core._env_flag("SKIP_GRAPH") or core._env_flag("SKIP_SEMANTIC"),
        "skip_wiki": core._env_flag("SKIP_WIKI"),
        "semantic_graph_enabled": semantic_enabled,
        "semantic_graph_reason": semantic_reason,
    }


def _source_snapshot_from_file_records(
    source_dir: str,
    file_records: Iterable[dict[str, int | str]],
) -> dict[str, Any]:
    core = _core()
    root = Path(source_dir).expanduser()
    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    dir_paths: set[Path] = {Path(".")}

    normalized_records: list[dict[str, Any]] = []
    for record in file_records:
        rel_path = core._normalize_source_path(str(record.get("path", "") or ""))
        if not rel_path:
            continue
        try:
            size = int(record.get("size", 0) or 0)
            mtime_ns = int(record.get("mtime_ns", 0) or 0)
        except (TypeError, ValueError):
            continue
        normalized_records.append(
            {
                "path": rel_path,
                "size": size,
                "mtime_ns": mtime_ns,
            }
        )

    normalized_records.sort(key=lambda item: str(item["path"]))
    for record in normalized_records:
        rel_path = str(record["path"])
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["size"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["mtime_ns"]).encode("utf-8"))
        digest.update(b"\0")
        files.append(record)

        parent = Path(rel_path).parent
        while True:
            normalized_parent = Path(".") if str(parent) in {"", "."} else parent
            dir_paths.add(normalized_parent)
            if normalized_parent == Path("."):
                break
            parent = normalized_parent.parent

    dirs: list[dict[str, Any]] = []
    for rel_dir in sorted(dir_paths, key=lambda path: path.as_posix()):
        dir_path = root if rel_dir == Path(".") else root / rel_dir
        try:
            stat = dir_path.stat()
        except OSError:
            continue
        rel_dir_text = "." if rel_dir == Path(".") else core._normalize_source_path(rel_dir.as_posix())
        dirs.append(
            {
                "path": rel_dir_text,
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )

    return {
        "version": 1,
        "file_count": len(files),
        "source_digest": digest.hexdigest(),
        "files": files,
        "dirs": dirs,
    }


def current_source_snapshot(source_dir: str) -> dict[str, Any]:
    core = _core()
    root = Path(source_dir).expanduser()
    file_records: list[dict[str, Any]] = []
    for path in core._iter_supported_files(root):
        try:
            stat = path.stat()
        except OSError:
            continue
        rel_path = core._normalize_source_path(path.relative_to(root).as_posix())
        if not rel_path:
            continue
        file_records.append(
            {
                "path": rel_path,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return _source_snapshot_from_file_records(source_dir, file_records)


def source_snapshot_from_indexed_files(
    indexed_files: list[Any],
    source_dir: str,
) -> dict[str, Any]:
    core = _core()
    root = Path(source_dir).expanduser()
    file_records: list[dict[str, Any]] = []
    for indexed in indexed_files:
        try:
            stat = indexed.file_path.stat()
        except OSError:
            continue
        try:
            rel_path = indexed.file_path.relative_to(root).as_posix()
        except ValueError:
            rel_path = indexed.rel_path
        normalized_rel_path = core._normalize_source_path(rel_path)
        if not normalized_rel_path:
            continue
        file_records.append(
            {
                "path": normalized_rel_path,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return _source_snapshot_from_file_records(source_dir, file_records)


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def stored_build_config_snapshot(
    previous_snapshot: dict[str, object],
    current_snapshot: dict[str, object],
) -> dict[str, object]:
    return {key: previous_snapshot.get(key) for key in current_snapshot}


def compare_stored_source_snapshot(
    source_dir: str,
    source_snapshot: dict[str, object],
) -> str:
    root = Path(source_dir).expanduser()
    raw_files = source_snapshot.get("files")
    raw_dirs = source_snapshot.get("dirs")
    if not isinstance(raw_files, list) or not isinstance(raw_dirs, list):
        return "invalid"

    dir_mismatch = False
    for entry in raw_dirs:
        if not isinstance(entry, dict):
            return "invalid"
        rel_dir = str(entry.get("path", "") or ".")
        dir_path = root if rel_dir in {"", "."} else root / rel_dir
        try:
            stat = dir_path.stat()
        except OSError:
            return "changed"
        try:
            expected_mtime = int(entry.get("mtime_ns", 0) or 0)
        except (TypeError, ValueError):
            return "invalid"
        if int(stat.st_mtime_ns) != expected_mtime:
            dir_mismatch = True

    for entry in raw_files:
        if not isinstance(entry, dict):
            return "invalid"
        rel_path = str(entry.get("path", "") or "")
        if not rel_path:
            return "invalid"
        file_path = root / rel_path
        try:
            stat = file_path.stat()
        except OSError:
            return "changed"
        try:
            expected_size = int(entry.get("size", 0) or 0)
            expected_mtime = int(entry.get("mtime_ns", 0) or 0)
        except (TypeError, ValueError):
            return "invalid"
        if int(stat.st_size) != expected_size or int(stat.st_mtime_ns) != expected_mtime:
            return "changed"

    return "dir_mismatch" if dir_mismatch else "match"


def source_snapshot_file_state(
    source_snapshot: dict[str, object],
) -> dict[str, object] | None:
    raw_files = source_snapshot.get("files")
    if not isinstance(raw_files, list):
        return None

    files: list[dict[str, object]] = []
    for entry in raw_files:
        if not isinstance(entry, dict):
            return None
        rel_path = str(entry.get("path", "") or "")
        if not rel_path:
            return None
        try:
            size = int(entry.get("size", 0) or 0)
            mtime_ns = int(entry.get("mtime_ns", 0) or 0)
        except (TypeError, ValueError):
            return None
        files.append(
            {
                "path": rel_path,
                "size": size,
                "mtime_ns": mtime_ns,
            }
        )

    try:
        file_count = int(source_snapshot.get("file_count", len(files)) or len(files))
    except (TypeError, ValueError):
        return None

    return {
        "file_count": file_count,
        "source_digest": str(source_snapshot.get("source_digest", "") or ""),
        "files": files,
    }
