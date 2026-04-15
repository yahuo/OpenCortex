#!/usr/bin/env python3
"""OpenCortex 启动器：重建索引并启动 Streamlit / FastAPI。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from ragbot import (
    FULLTEXT_INDEX_FILENAME,
    build_vectorstore,
    current_build_config_snapshot,
    current_build_snapshot,
    current_source_snapshot,
)
from ragbot_runtime import load_fulltext_index_payload

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_RUNTIME_INDEX_ARTIFACTS = ("index.faiss", "index.pkl", "index_manifest.json")


def read_runtime_config() -> dict[str, str]:
    source_dir = os.getenv("LOCAL_DOCS_DIR", "").strip() or "./docs"
    host = os.getenv("APP_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = os.getenv("APP_PORT", "8501").strip() or "8501"
    return {
        "source_dir": str(Path(source_dir).expanduser()),
        "persist_dir": str(
            Path(os.getenv("CHROMA_PERSIST_DIR", "~/wechat_rag_db")).expanduser()
        ),
        "embed_api_key": os.getenv("EMBED_API_KEY", "").strip(),
        "embed_base_url": os.getenv(
            "EMBED_BASE_URL", "https://api.siliconflow.cn/v1"
        ).strip(),
        "embed_model": os.getenv("EMBED_MODEL", "BAAI/bge-m3").strip(),
        "llm_api_key": os.getenv("LLM_API_KEY", "").strip(),
        "llm_base_url": os.getenv(
            "LLM_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ).strip(),
        "llm_model": os.getenv("LLM_MODEL", "gemini-2.0-flash").strip(),
        "host": host,
        "port": port,
        "api_port": os.getenv("API_PORT", "8502").strip() or "8502",
    }


def validate_runtime(cfg: dict[str, str]) -> None:
    if not cfg["port"].isdigit():
        raise ValueError(f"APP_PORT 非法: {cfg['port']}")
    port = int(cfg["port"])
    if not (1 <= port <= 65535):
        raise ValueError(f"APP_PORT 超出范围: {cfg['port']}")
    if not cfg["api_port"].isdigit():
        raise ValueError(f"API_PORT 非法: {cfg['api_port']}")
    api_port = int(cfg["api_port"])
    if not (1 <= api_port <= 65535):
        raise ValueError(f"API_PORT 超出范围: {cfg['api_port']}")


def validate_rebuild_runtime(cfg: dict[str, str]) -> None:
    if not cfg["embed_api_key"]:
        raise ValueError("缺少环境变量 EMBED_API_KEY")
    if not Path(cfg["source_dir"]).is_dir():
        raise ValueError(f"源目录不存在: {cfg['source_dir']}")


def ensure_index_artifacts(cfg: dict[str, str]) -> None:
    required = runtime_index_artifacts(cfg)
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise ValueError(
            "缺少索引文件，请先执行 `python start.py --rebuild-only`："
            + ", ".join(missing)
        )
    manifest = load_index_manifest(cfg)
    if manifest is None:
        raise ValueError("索引 manifest 无法解析，请先执行 `python start.py --rebuild-only`。")
    validate_fulltext_index_artifact(cfg, manifest=manifest)


def load_index_manifest(cfg: dict[str, str]) -> dict | None:
    manifest_path = Path(cfg["persist_dir"]) / "index_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def runtime_index_artifacts(cfg: dict[str, str], manifest: dict | None = None) -> list[Path]:
    persist_dir = Path(cfg["persist_dir"])
    artifacts = [persist_dir / name for name in BASE_RUNTIME_INDEX_ARTIFACTS]
    resolved_manifest = manifest if manifest is not None else load_index_manifest(cfg)
    fulltext_relpath = FULLTEXT_INDEX_FILENAME
    if isinstance(resolved_manifest, dict):
        fulltext_relpath = str(
            resolved_manifest.get("fulltext_index_file", FULLTEXT_INDEX_FILENAME)
            or FULLTEXT_INDEX_FILENAME
        )
    artifacts.append(persist_dir / Path(fulltext_relpath))
    return artifacts


def fulltext_index_path(cfg: dict[str, str], manifest: dict | None = None) -> Path:
    return runtime_index_artifacts(cfg, manifest=manifest)[-1]


def validate_fulltext_index_artifact(cfg: dict[str, str], manifest: dict | None = None) -> None:
    load_fulltext_index_payload(fulltext_index_path(cfg, manifest=manifest))


def build_config_snapshot(cfg: dict[str, str]) -> dict[str, object]:
    return current_build_config_snapshot(
        cfg["source_dir"],
        embed_base_url=cfg["embed_base_url"],
        embed_model=cfg["embed_model"],
        llm_api_key=cfg["llm_api_key"],
        llm_model=cfg["llm_model"],
        llm_base_url=cfg["llm_base_url"],
    )


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


def index_rebuild_state(cfg: dict[str, str]) -> tuple[bool, str]:
    persist_dir = Path(cfg["persist_dir"])
    required = [persist_dir / name for name in BASE_RUNTIME_INDEX_ARTIFACTS]
    if any(not path.exists() for path in required):
        return True, "missing_index"

    manifest = load_index_manifest(cfg)
    if manifest is None:
        return True, "invalid_manifest"
    required = runtime_index_artifacts(cfg, manifest=manifest)
    if any(not path.exists() for path in required):
        return True, "missing_index"
    try:
        validate_fulltext_index_artifact(cfg, manifest=manifest)
    except ValueError:
        return True, "invalid_fulltext_index"

    previous_snapshot = manifest.get("build_snapshot")
    if not isinstance(previous_snapshot, dict):
        return True, "missing_build_snapshot"

    current_snapshot = build_config_snapshot(cfg)
    if stored_build_config_snapshot(previous_snapshot, current_snapshot) != current_snapshot:
        return True, "build_config_changed"

    previous_source_snapshot = manifest.get("source_snapshot")
    if isinstance(previous_source_snapshot, dict):
        source_state = compare_stored_source_snapshot(
            cfg["source_dir"],
            previous_source_snapshot,
        )
        if source_state == "match":
            return False, "up_to_date"
        if source_state == "dir_mismatch":
            previous_file_state = source_snapshot_file_state(previous_source_snapshot)
            current_file_state = source_snapshot_file_state(
                current_source_snapshot(cfg["source_dir"])
            )
            if previous_file_state is None or current_file_state is None:
                return True, "invalid_source_snapshot"
            if current_file_state != previous_file_state:
                return True, "source_changed"
            return False, "up_to_date"
        if source_state == "changed":
            return True, "source_changed"
        return True, "invalid_source_snapshot"

    if {"file_count", "source_digest"}.issubset(previous_snapshot):
        current_full_snapshot = current_build_snapshot(
            cfg["source_dir"],
            embed_base_url=cfg["embed_base_url"],
            embed_model=cfg["embed_model"],
            llm_api_key=cfg["llm_api_key"],
            llm_model=cfg["llm_model"],
            llm_base_url=cfg["llm_base_url"],
        )
        if previous_snapshot != current_full_snapshot:
            return True, "source_changed"
        return False, "up_to_date"

    return True, "missing_source_snapshot"


def rebuild_reason_message(reason: str) -> str:
    messages = {
        "missing_index": "未找到可用索引，开始首次构建。",
        "invalid_manifest": "索引 manifest 无法解析，开始重建。",
        "invalid_fulltext_index": "全文索引不可用，开始重建。",
        "missing_build_snapshot": "旧索引缺少构建快照，开始重建。",
        "missing_source_snapshot": "旧索引缺少文档快照，开始重建。",
        "invalid_source_snapshot": "旧索引的文档快照无效，开始重建。",
        "build_config_changed": "检测到构建配置变化，开始重建索引。",
        "source_changed": "检测到文档变化，开始重建索引。",
    }
    return messages.get(reason, "索引状态未知，开始重建。")


def rebuild_index(cfg: dict[str, str]) -> None:
    print("[OpenCortex] 开始重建向量索引...", flush=True)

    def on_progress(current: int, total: int, message: str) -> None:
        percent = 0 if total <= 0 else min(int(current * 100 / total), 100)
        print(f"[OpenCortex][{percent:>3}%] {message}", flush=True)

    build_vectorstore(
        md_dir=cfg["source_dir"],
        embed_api_key=cfg["embed_api_key"],
        embed_base_url=cfg["embed_base_url"],
        embed_model=cfg["embed_model"],
        persist_dir=cfg["persist_dir"],
        progress_callback=on_progress,
        llm_api_key=cfg["llm_api_key"],
        llm_model=cfg["llm_model"],
        llm_base_url=cfg["llm_base_url"],
    )


def launch_streamlit(cfg: dict[str, str]) -> subprocess.Popen:
    """启动 Streamlit Web 服务（后台子进程）"""
    url = f"http://{cfg['host']}:{cfg['port']}"
    print(f"[OpenCortex] Web 服务启动中: {url}", flush=True)

    app_path = PROJECT_ROOT / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless",
        "true",
        "--server.address",
        cfg["host"],
        "--server.port",
        cfg["port"],
    ]
    env = os.environ.copy()
    env["BROWSER"] = "none"
    return subprocess.Popen(cmd, env=env, cwd=str(PROJECT_ROOT))


def launch_api(cfg: dict[str, str]) -> subprocess.Popen:
    """启动 FastAPI API 服务（后台子进程）"""
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api:app",
        "--host",
        cfg["host"],
        "--port",
        cfg["api_port"],
    ]
    print(
        f"[OpenCortex] API  服务启动中: http://{cfg['host']}:{cfg['api_port']}",
        flush=True,
    )
    return subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenCortex 启动器")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--rebuild-only",
        action="store_true",
        help="仅重建向量索引，不启动 Streamlit",
    )
    mode_group.add_argument(
        "--skip-rebuild",
        action="store_true",
        help="跳过索引重建，直接启动 Streamlit 和 FastAPI",
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    cfg = read_runtime_config()
    rebuilt = False
    skipped_rebuild = False

    try:
        validate_runtime(cfg)
        if args.skip_rebuild:
            ensure_index_artifacts(cfg)
            skipped_rebuild = True
        elif args.rebuild_only:
            validate_rebuild_runtime(cfg)
            rebuild_index(cfg)
            rebuilt = True
        else:
            needs_rebuild, reason = index_rebuild_state(cfg)
            if needs_rebuild:
                validate_rebuild_runtime(cfg)
                print(f"[OpenCortex] {rebuild_reason_message(reason)}", flush=True)
                rebuild_index(cfg)
                rebuilt = True
            else:
                print("[OpenCortex] 索引未变化，跳过重建。", flush=True)
                skipped_rebuild = True
    except Exception as exc:
        print(f"[OpenCortex] 启动失败: {exc}", flush=True)
        return 1

    if args.rebuild_only:
        print("[OpenCortex] 索引重建完成，退出（--rebuild-only 模式）。", flush=True)
        return 0

    if skipped_rebuild:
        print("[OpenCortex] 已跳过索引重建，直接启动服务。", flush=True)
    elif rebuilt:
        print(
            "[OpenCortex] 索引重建完成，正在启动服务（不会自动打开浏览器）...",
            flush=True,
        )

    procs = [launch_streamlit(cfg), launch_api(cfg)]
    try:
        while True:
            for p in procs:
                ret = p.poll()
                if ret is not None:
                    for other in procs:
                        other.terminate()
                    return ret
            time.sleep(0.5)
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
