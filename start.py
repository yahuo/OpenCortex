#!/usr/bin/env python3
"""OpenCortex 启动器：重建索引并启动 Streamlit / FastAPI。"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from ragbot import build_vectorstore

PROJECT_ROOT = Path(__file__).resolve().parent


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
    persist_dir = Path(cfg["persist_dir"])
    required = [persist_dir / "index.faiss", persist_dir / "index_manifest.json"]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise ValueError(
            "缺少索引文件，请先执行 `python start.py --rebuild-only`："
            + ", ".join(missing)
        )


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

    try:
        validate_runtime(cfg)
        if args.skip_rebuild:
            ensure_index_artifacts(cfg)
        else:
            validate_rebuild_runtime(cfg)
            rebuild_index(cfg)
    except Exception as exc:
        print(f"[OpenCortex] 启动失败: {exc}", flush=True)
        return 1

    if args.rebuild_only:
        print("[OpenCortex] 索引重建完成，退出（--rebuild-only 模式）。", flush=True)
        return 0

    if args.skip_rebuild:
        print("[OpenCortex] 已跳过索引重建，直接启动服务。", flush=True)
    else:
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
