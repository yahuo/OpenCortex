#!/usr/bin/env python3
"""OpenCortex 启动器：先重建索引，再启动 Streamlit。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import argparse

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
        "host": host,
        "port": port,
    }


def validate_runtime(cfg: dict[str, str]) -> None:
    if not cfg["embed_api_key"]:
        raise ValueError("缺少环境变量 EMBED_API_KEY")
    if not Path(cfg["source_dir"]).is_dir():
        raise ValueError(f"源目录不存在: {cfg['source_dir']}")
    if not cfg["port"].isdigit():
        raise ValueError(f"APP_PORT 非法: {cfg['port']}")
    port = int(cfg["port"])
    if not (1 <= port <= 65535):
        raise ValueError(f"APP_PORT 超出范围: {cfg['port']}")


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
    )


def launch_streamlit(cfg: dict[str, str]) -> int:
    url = f"http://{cfg['host']}:{cfg['port']}"
    print(f"[OpenCortex] 索引重建完成。请访问: {url}", flush=True)
    print("[OpenCortex] 正在启动 Web 服务（不会自动打开浏览器）...", flush=True)

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

    proc = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenCortex 启动器")
    parser.add_argument(
        "--rebuild-only",
        action="store_true",
        help="仅重建向量索引，不启动 Streamlit",
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    cfg = read_runtime_config()

    try:
        validate_runtime(cfg)
        rebuild_index(cfg)
    except Exception as exc:
        print(f"[OpenCortex] 启动失败: {exc}", flush=True)
        return 1

    if args.rebuild_only:
        print("[OpenCortex] 索引重建完成，退出（--rebuild-only 模式）。", flush=True)
        return 0

    return launch_streamlit(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
