import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import streamlit as st

import export_group as eg
from decrypt_wechat_db import batch_decrypt

# 加载 .env 配置
load_dotenv()

st.set_page_config(page_title="WechatLLM 微信知识库导出工具", page_icon="💬", layout="wide")

# ─────────────────────────────────────────
# 🎨 全局 CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* 主标题渐变 */
.hero-title {
    background: linear-gradient(135deg, #4ade80 0%, #22d3ee 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 0.15rem;
}
.hero-sub { color: #64748b; font-size: 0.85rem; margin-bottom: 0; }

/* 侧边栏卡片区块 */
.sidebar-section {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1rem 0.8rem;
    margin-bottom: 1rem;
}
.sidebar-section-title {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #4ade80;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* 状态栏 */
.status-bar {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    font-size: 0.72rem;
    color: #64748b;
    font-family: monospace;
    line-height: 1.7;
    margin-top: 0.5rem;
    word-break: break-all;
}
.status-bar span { color: #38bdf8; }

/* 隐藏默认元素 */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* 内容区 */
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; }

/* 主按钮 */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4ade80, #22d3ee);
    color: #0f172a;
    border: none;
    font-weight: 600;
    border-radius: 10px;
    transition: all 0.2s ease;
    box-shadow: 0 2px 12px rgba(74,222,128,0.3);
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(74,222,128,0.45);
}
.stButton > button[kind="primary"]:disabled {
    opacity: 0.35;
    transform: none;
    box-shadow: none;
}

/* 输入框 */
.stTextInput > div > div > input {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    color: #e2e8f0;
    font-size: 0.85rem;
}
.stTextInput > div > div > input:focus {
    border-color: #4ade80;
    box-shadow: 0 0 0 2px rgba(74,222,128,0.2);
}

/* 右侧主区域主标题行 */
.main-header {
    display: flex;
    align-items: baseline;
    gap: 0.8rem;
    margin-bottom: 0.8rem;
}
.contact-count {
    font-size: 0.82rem;
    color: #64748b;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# 核心逻辑
# ─────────────────────────────────────────
@st.cache_data
def load_all_contacts(contact_db_path: str) -> pd.DataFrame:
    if not os.path.exists(contact_db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(contact_db_path)
    try:
        df = pd.read_sql_query("SELECT username, nick_name, remark FROM contact", conn)
    finally:
        conn.close()
    df["Display Name"] = df["remark"].where(df["remark"] != "", df["nick_name"])
    df["Type"] = df["username"].apply(lambda x: "🌐 群聊" if "@chatroom" in x else "👤 私聊")
    df = df[df["Display Name"].notna() & (df["Display Name"] != "")]
    return df[["Display Name", "Type", "username"]].copy()


def sync_runtime_paths(root_base=None, db_dir=None, out_dir=None):
    eg.refresh_runtime_config(root_base=root_base, db_dir=db_dir, out_dir=out_dir)
    load_all_contacts.clear()


cfg = eg.refresh_runtime_config()
default_work_dir = (
    os.path.dirname(cfg["DB_DIR"]) if os.path.basename(cfg["DB_DIR"]) == "db_storage" else cfg["DB_DIR"]
)

st.session_state.setdefault("decrypt_data_dir", cfg["ROOT_BASE"])
st.session_state.setdefault("decrypt_work_dir", default_work_dir)
st.session_state.setdefault("decrypt_key", "")
st.session_state.setdefault("markdown_out_dir", cfg["OUT_DIR"])


# ═══════════════════════════════════════════════════
# 左侧边栏：所有配置项
# ═══════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="hero-title">💬 WechatLLM</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">微信知识库导出控制台</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── 解密配置区（可折叠）──
    with st.expander("🔓 解密数据库", expanded=True):
        decrypt_key = st.text_input(
            "数据库密钥 (64 hex)",
            value=st.session_state.decrypt_key,
            type="password",
            placeholder="a639...08342",
        ).strip()

        detected = eg.detect_wechat_account_dirs()
        if detected:
            dropdown_options = {name: path for name, path in detected}
            selected_account = st.selectbox(
                "微信账号目录",
                options=list(dropdown_options.keys()),
                format_func=lambda x: f"📱 {x}",
                help="自动扫描得到的有效账号目录",
            )
            if selected_account:
                auto_path = dropdown_options[selected_account]
                if st.session_state.decrypt_data_dir != auto_path:
                    st.session_state.decrypt_data_dir = auto_path
        else:
            st.caption("⚠️ 未自动检测到微信账号目录，请手动填写。")

        decrypt_data_dir = st.text_input(
            "微信账号数据目录",
            value=st.session_state.decrypt_data_dir,
            placeholder="/Users/.../xwechat_files/<账号目录>",
        ).strip()

        decrypt_work_dir = st.text_input(
            "解密输出目录",
            value=st.session_state.decrypt_work_dir,
            placeholder="/Users/.../wechat_export_test",
        ).strip()

        st.session_state.decrypt_key = decrypt_key
        st.session_state.decrypt_data_dir = decrypt_data_dir
        st.session_state.decrypt_work_dir = decrypt_work_dir

        run_decrypt = st.button("🔑 开始解密", type="primary", use_container_width=True)

    # ── 导出目录配置（可折叠）──
    with st.expander("📁 导出目录", expanded=False):
        markdown_out_dir = st.text_input(
            "Markdown 输出目录",
            value=st.session_state.markdown_out_dir,
            placeholder="/Users/.../wechat_export",
            label_visibility="collapsed",
        ).strip()

        apply_out_dir = st.button("✅ 应用目录", use_container_width=True)

        if markdown_out_dir:
            st.session_state.markdown_out_dir = markdown_out_dir

    # ── RAG 知识库配置（可折叠）──
    with st.expander("🧠 RAG 问答配置", expanded=False):
        st.caption("Embedding 模型")
        embed_api_key = st.text_input(
            "Embedding API Key",
            value=os.getenv("EMBED_API_KEY", ""),
            type="password",
            placeholder="sf-xxx",
        ).strip()
        embed_base_url = st.text_input(
            "Embedding Base URL",
            value=os.getenv("EMBED_BASE_URL", "https://api.siliconflow.cn/v1"),
            placeholder="https://api.siliconflow.cn/v1",
        ).strip()
        embed_model = st.text_input(
            "Embedding 模型名称",
            value=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
            placeholder="BAAI/bge-m3",
        ).strip()

        st.caption("LLM 模型")
        llm_api_key = st.text_input(
            "LLM API Key",
            value=os.getenv("LLM_API_KEY", ""),
            type="password",
            placeholder="AIza-xxx 或 sk-xxx",
        ).strip()
        llm_base_url = st.text_input(
            "LLM Base URL",
            value=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
            placeholder="https://api.moonshot.cn/v1",
        ).strip()
        llm_model = st.text_input(
            "LLM 模型名称",
            value=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
            placeholder="gemini-2.0-flash / moonshot-v1-32k",
        ).strip()

        chroma_dir = str(Path(os.getenv("CHROMA_PERSIST_DIR", "~/wechat_rag_db")).expanduser())
        st.caption(f"📦 ChromaDB: `{chroma_dir}`")

    # ── 状态栏 ──
    st.markdown(f"""
<div class="status-bar">
  DB: <span>{eg.DB_DIR}</span><br>
  OUT: <span>{eg.OUT_DIR}</span>
</div>
""", unsafe_allow_html=True)



# ═══════════════════════════════════════════════════
# 处理侧边栏事件（必须在 sidebar with 块外面）
# ═══════════════════════════════════════════════════
if run_decrypt:
    if not decrypt_key:
        st.sidebar.error("⚠️ 请填写数据库密钥。")
    elif not os.path.isdir(decrypt_data_dir):
        st.sidebar.error(f"⚠️ 目录不存在: {decrypt_data_dir}")
    elif not decrypt_work_dir:
        st.sidebar.error("⚠️ 请填写解密输出目录。")
    else:
        try:
            with st.sidebar.status("⚙️ 正在解密数据库..."):
                stats, failures = batch_decrypt(decrypt_data_dir, decrypt_work_dir, decrypt_key)
        except Exception as exc:
            st.sidebar.error(f"❌ 解密失败: {exc}")
            stats, failures = None, None

        if stats is not None:
            decrypted_db_dir = os.path.join(decrypt_work_dir, "db_storage")
            if os.path.isdir(decrypted_db_dir):
                sync_runtime_paths(
                    root_base=decrypt_data_dir,
                    db_dir=decrypted_db_dir,
                    out_dir=st.session_state.markdown_out_dir,
                )
            if stats.failed == 0:
                st.sidebar.success(f"✅ 成功解密 {stats.success}/{stats.total} 个数据库")
            else:
                st.sidebar.warning(f"⚠️ 成功 {stats.success}，失败 {stats.failed}")

if apply_out_dir:
    if not markdown_out_dir:
        st.sidebar.error("❌ 输出目录不能为空。")
    else:
        sync_runtime_paths(out_dir=markdown_out_dir)
        st.sidebar.success(f"✅ 已应用: {eg.OUT_DIR}")


# ═══════════════════════════════════════════════════
# 右侧主区域：联系人列表与批量导出
# ═══════════════════════════════════════════════════
df_contacts = load_all_contacts(eg.CONTACT_DB)

if df_contacts.empty:
    st.markdown('<div class="hero-title">💬 WechatLLM</div>', unsafe_allow_html=True)
    st.info("🔍 未检测到联系人库，请先在左侧完成解密配置。")
else:
    total_count = len(df_contacts)

    col_search, col_btn = st.columns([5, 1])
    with col_search:
        search_query = st.text_input(
            "search",
            placeholder="🔍  搜索联系人或群聊名称...",
            label_visibility="collapsed",
        ).strip()

    progress_container = st.empty()
    status_text = st.empty()

    if search_query:
        filtered_df = df_contacts[
            df_contacts["Display Name"].str.contains(search_query, case=False, na=False)
        ].copy()
    else:
        filtered_df = df_contacts.copy()

    st.caption(f"共 {total_count} 个联系人 / 群聊，当前显示 {len(filtered_df)} 条")

    filtered_df.insert(0, "Export", False)

    edited_df = st.data_editor(
        filtered_df,
        column_config={
            "Export": st.column_config.CheckboxColumn("导出", default=False),
            "Display Name": st.column_config.TextColumn("对话名称", width="large"),
            "Type": st.column_config.TextColumn("类型", width="small"),
            "username": st.column_config.TextColumn("内部 ID", width="medium"),
        },
        disabled=["Display Name", "Type", "username"],
        hide_index=True,
        use_container_width=True,
    )

    selected_rows = edited_df[edited_df["Export"] == True]

    with col_btn:
        start_export = st.button(
            "🚀 导出选定",
            type="primary",
            use_container_width=True,
            disabled=selected_rows.empty,
        )

    if not selected_rows.empty:
        st.success(f"✅ 已选中 **{len(selected_rows)}** 个会话，点击右上角按钮批量导出。")

    if start_export and not selected_rows.empty:
        my_bar = progress_container.progress(0, text="准备开始...")
        status_text.info("⚙️ 正在批量导出，请稍候...")
        success_count = 0

        for _, row in selected_rows.iterrows():
            display_name = row["Display Name"]
            username = row["username"]
            st.toast(f"🔄 {display_name}")

            def on_progress(current, total, msg, _name=display_name):
                percent = current / total if total > 0 else 0
                my_bar.progress(percent, text=f"[{_name}] {msg}")

            try:
                success, reason_or_file = eg.export_by_username(
                    username, display_name, progress_callback=on_progress
                )
                if success:
                    success_count += 1
                else:
                    st.error(f"❌ {display_name}: {reason_or_file}")
            except Exception as exc:
                st.error(f"💥 {display_name}: {exc}")

        my_bar.empty()
        if success_count == len(selected_rows):
            status_text.success(f"🎉 全部完成！成功导出 **{success_count}** 个文件到 `{eg.OUT_DIR}`")
            st.balloons()
        else:
            status_text.warning(f"完成。成功 {success_count}，失败 {len(selected_rows) - success_count}。")


# ═══════════════════════════════════════════════════
# 🧠 RAG 知识库问答
# ═══════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="sidebar-section-title" style="font-size:1rem;margin-bottom:0.8rem">🧠 知识库问答</div>', unsafe_allow_html=True)

if not embed_api_key or not llm_api_key:
    st.info("🔑 请先展开左侧『RAG 问答配置』填写 Embedding Key 和 LLM Key。")
else:
    from ragbot import build_vectorstore, load_vectorstore, ask as rag_ask

    col_idx_btn, col_idx_status = st.columns([2, 5])
    with col_idx_btn:
        build_btn = st.button("🔧 构建 / 更新索引", use_container_width=True)
    with col_idx_status:
        md_out = eg.OUT_DIR
        st.caption(f"建索目录：`{md_out}` → ChromaDB：`{chroma_dir}`")

    if build_btn:
        with st.status("正在构建向量索引...", expanded=True) as build_status:
            def idx_progress(cur, total, msg):
                st.write(f"{msg}")
            try:
                build_vectorstore(
                    md_dir=md_out,
                    embed_api_key=embed_api_key,
                    embed_base_url=embed_base_url,
                    embed_model=embed_model,
                    persist_dir=chroma_dir,
                    progress_callback=idx_progress,
                )
                build_status.update(label="✅ 索引构建完成！", state="complete")
                st.session_state["vectorstore_ready"] = True
            except Exception as exc:
                build_status.update(label=f"❌ 构建失败: {exc}", state="error")

    # 加载或检测已有索引
    vectorstore = None
    if st.session_state.get("vectorstore_ready") or Path(chroma_dir).exists():
        try:
            vectorstore = load_vectorstore(
                embed_api_key=embed_api_key,
                embed_base_url=embed_base_url,
                embed_model=embed_model,
                persist_dir=chroma_dir,
            )
            if vectorstore:
                st.session_state["vectorstore_ready"] = True
        except Exception:
            pass

    if vectorstore is None:
        st.info("📦 未检测到向量索引，请先导出 MD 文件后点击“构建 / 更新索引”。")
    else:
        # 聊天历史
        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = []

        for msg in st.session_state.rag_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📎 查看引用来源"):
                        for src in msg["sources"]:
                            st.markdown(f"**📄 {src['source']}** `{src['time_range']}`")
                            st.caption(src["snippet"])

        if question := st.chat_input("输入问题，例如：上次技术讨论了什么？"):
            st.session_state.rag_messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("💭 正在检索记忆中..."):
                    try:
                        result = rag_ask(
                            question=question,
                            vectorstore=vectorstore,
                            llm_api_key=llm_api_key,
                            llm_model=llm_model,
                            llm_base_url=llm_base_url,
                        )
                        answer = result["answer"]
                        sources = result["sources"]
                    except Exception as exc:
                        answer = f"❌ 问答失败: {exc}"
                        sources = []

                st.markdown(answer)
                if sources:
                    with st.expander("📎 查看引用来源"):
                        for src in sources:
                            st.markdown(f"**📄 {src['source']}** `{src['time_range']}`")
                            st.caption(src["snippet"])

            st.session_state.rag_messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
