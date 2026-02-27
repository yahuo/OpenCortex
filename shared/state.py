"""session_state 初始化与跨页面共享的工具函数"""
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import export_group as eg

load_dotenv()


def init_defaults():
    """初始化所有 session_state 默认值（仅首次运行时生效）"""
    cfg = eg.refresh_runtime_config()
    default_work_dir = (
        os.path.dirname(cfg["DB_DIR"])
        if os.path.basename(cfg["DB_DIR"]) == "db_storage"
        else cfg["DB_DIR"]
    )

    # 解密相关
    st.session_state.setdefault("decrypt_data_dir", cfg["ROOT_BASE"])
    st.session_state.setdefault("decrypt_work_dir", default_work_dir)
    st.session_state.setdefault("decrypt_key", "")

    # 导出目录
    st.session_state.setdefault("markdown_out_dir", cfg["OUT_DIR"])

    # RAG 配置（初始值从 .env 读取，配置页修改后通过 widget key 自动持久化）
    st.session_state.setdefault(
        "embed_api_key", os.getenv("EMBED_API_KEY", "")
    )
    st.session_state.setdefault(
        "embed_base_url",
        os.getenv("EMBED_BASE_URL", "https://api.siliconflow.cn/v1"),
    )
    st.session_state.setdefault(
        "embed_model", os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    )
    st.session_state.setdefault(
        "llm_api_key", os.getenv("LLM_API_KEY", "")
    )
    st.session_state.setdefault(
        "llm_base_url",
        os.getenv(
            "LLM_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
    )
    st.session_state.setdefault(
        "llm_model", os.getenv("LLM_MODEL", "gemini-2.0-flash")
    )
    st.session_state.setdefault(
        "chroma_dir",
        str(Path(os.getenv("CHROMA_PERSIST_DIR", "~/wechat_rag_db")).expanduser()),
    )

    # 向量索引状态
    st.session_state.setdefault("vectorstore_ready", False)

    # 聊天历史
    st.session_state.setdefault("rag_messages", [])

    # 索引文件清单元数据
    st.session_state.setdefault("indexed_files", [])
    st.session_state.setdefault("index_build_time", "")


def sync_runtime_paths(root_base=None, db_dir=None, out_dir=None):
    """更新运行时路径配置并清除联系人缓存"""
    eg.refresh_runtime_config(root_base=root_base, db_dir=db_dir, out_dir=out_dir)
    load_all_contacts.clear()


@st.cache_data
def load_all_contacts(contact_db_path: str) -> pd.DataFrame:
    """加载联系人列表（带缓存）"""
    if not os.path.exists(contact_db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(contact_db_path)
    try:
        df = pd.read_sql_query(
            "SELECT username, nick_name, remark FROM contact", conn
        )
    finally:
        conn.close()
    df["Display Name"] = df["remark"].where(df["remark"] != "", df["nick_name"])
    df["Type"] = df["username"].apply(
        lambda x: "🌐 群聊" if "@chatroom" in x else "👤 私聊"
    )
    df = df[df["Display Name"].notna() & (df["Display Name"] != "")]
    return df[["Display Name", "Type", "username"]].copy()
