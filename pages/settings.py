"""配置与管理页面 — 解密、导出、RAG 配置、索引管理"""
import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

import export_group as eg
from decrypt_wechat_db import batch_decrypt
from shared.state import sync_runtime_paths, load_all_contacts

st.markdown("## ⚙️ 配置与管理")

tab_decrypt, tab_export, tab_rag, tab_index = st.tabs(
    ["🔓 解密数据库", "📁 联系人导出", "🧠 RAG 配置", "📦 索引管理"]
)

# ═══════════════════════════════════════════════════
# Tab 1: 解密数据库
# ═══════════════════════════════════════════════════
with tab_decrypt:
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

    if st.button("🔑 开始解密", type="primary", use_container_width=True):
        if not decrypt_key:
            st.error("⚠️ 请填写数据库密钥。")
        elif not os.path.isdir(decrypt_data_dir):
            st.error(f"⚠️ 目录不存在: {decrypt_data_dir}")
        elif not decrypt_work_dir:
            st.error("⚠️ 请填写解密输出目录。")
        else:
            try:
                with st.status("⚙️ 正在解密数据库..."):
                    stats, failures = batch_decrypt(
                        decrypt_data_dir, decrypt_work_dir, decrypt_key
                    )
            except Exception as exc:
                st.error(f"❌ 解密失败: {exc}")
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
                    st.success(
                        f"✅ 成功解密 {stats.success}/{stats.total} 个数据库"
                    )
                else:
                    st.warning(f"⚠️ 成功 {stats.success}，失败 {stats.failed}")

# ═══════════════════════════════════════════════════
# Tab 2: 联系人导出
# ═══════════════════════════════════════════════════
with tab_export:
    # 导出目录配置
    markdown_out_dir = st.text_input(
        "Markdown 输出目录",
        value=st.session_state.markdown_out_dir,
        placeholder="/Users/.../wechat_export",
    ).strip()

    if st.button("✅ 应用目录", use_container_width=False):
        if not markdown_out_dir:
            st.error("❌ 输出目录不能为空。")
        else:
            st.session_state.markdown_out_dir = markdown_out_dir
            sync_runtime_paths(out_dir=markdown_out_dir)
            st.success(f"✅ 已应用: {eg.OUT_DIR}")

    if markdown_out_dir:
        st.session_state.markdown_out_dir = markdown_out_dir

    st.markdown("---")

    # 联系人表格
    df_contacts = load_all_contacts(eg.CONTACT_DB)

    if df_contacts.empty:
        st.info("🔍 未检测到联系人库，请先在『解密数据库』完成解密配置。")
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
                df_contacts["Display Name"].str.contains(
                    search_query, case=False, na=False
                )
            ].copy()
        else:
            filtered_df = df_contacts.copy()

        st.caption(
            f"共 {total_count} 个联系人 / 群聊，当前显示 {len(filtered_df)} 条"
        )

        filtered_df.insert(0, "Export", False)

        edited_df = st.data_editor(
            filtered_df,
            column_config={
                "Export": st.column_config.CheckboxColumn("导出", default=False),
                "Display Name": st.column_config.TextColumn(
                    "对话名称", width="large"
                ),
                "Type": st.column_config.TextColumn("类型", width="small"),
                "username": st.column_config.TextColumn("内部 ID", width="medium"),
            },
            disabled=["Display Name", "Type", "username"],
            hide_index=True,
            use_container_width=True,
        )

        selected_rows = edited_df[edited_df["Export"] == True]  # noqa: E712

        with col_btn:
            start_export = st.button(
                "🚀 导出选定",
                type="primary",
                use_container_width=True,
                disabled=selected_rows.empty,
            )

        if not selected_rows.empty:
            st.success(
                f"✅ 已选中 **{len(selected_rows)}** 个会话，点击右上角按钮批量导出。"
            )

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
                status_text.success(
                    f"🎉 全部完成！成功导出 **{success_count}** 个文件到 `{eg.OUT_DIR}`"
                )
                st.balloons()
            else:
                status_text.warning(
                    f"完成。成功 {success_count}，失败 {len(selected_rows) - success_count}。"
                )

# ═══════════════════════════════════════════════════
# Tab 3: RAG 配置
# ═══════════════════════════════════════════════════
with tab_rag:
    st.caption("Embedding 模型")
    st.text_input(
        "Embedding API Key",
        type="password",
        placeholder="sf-xxx",
        key="embed_api_key",
    )
    st.text_input(
        "Embedding Base URL",
        placeholder="https://api.siliconflow.cn/v1",
        key="embed_base_url",
    )
    st.text_input(
        "Embedding 模型名称",
        placeholder="BAAI/bge-m3",
        key="embed_model",
    )

    st.markdown("---")
    st.caption("LLM 模型")
    st.text_input(
        "LLM API Key",
        type="password",
        placeholder="AIza-xxx 或 sk-xxx",
        key="llm_api_key",
    )
    st.text_input(
        "LLM Base URL",
        placeholder="https://api.moonshot.cn/v1",
        key="llm_base_url",
    )
    st.text_input(
        "LLM 模型名称",
        placeholder="gemini-2.0-flash / moonshot-v1-32k",
        key="llm_model",
    )

    st.markdown("---")
    st.caption(f"📦 FAISS 索引目录: `{st.session_state.chroma_dir}`")

# ═══════════════════════════════════════════════════
# Tab 4: 索引管理
# ═══════════════════════════════════════════════════

def _load_manifest(persist_dir: str):
    """从 index_manifest.json 加载已索引文件清单到 session_state"""
    manifest_path = Path(persist_dir) / "index_manifest.json"
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            st.session_state.indexed_files = data.get("files", [])
            st.session_state.index_build_time = data.get("build_time", "")
            return
        except Exception:
            pass

    # 回退：尝试从 FAISS docstore 提取 source 字段
    index_file = Path(persist_dir) / "index.faiss"
    if not index_file.exists():
        st.session_state.indexed_files = []
        st.session_state.index_build_time = ""
        return

    try:
        from ragbot import load_vectorstore

        vs = load_vectorstore(
            embed_api_key=st.session_state.get("embed_api_key", ""),
            embed_base_url=st.session_state.get("embed_base_url", ""),
            embed_model=st.session_state.get("embed_model", ""),
            persist_dir=persist_dir,
        )
        if vs and hasattr(vs, "docstore") and hasattr(vs.docstore, "_dict"):
            sources: dict[str, dict] = {}
            for doc in vs.docstore._dict.values():
                src = doc.metadata.get("source", "")
                if src:
                    if src not in sources:
                        sources[src] = {"name": src, "chunks": 0}
                    sources[src]["chunks"] += 1
            st.session_state.indexed_files = sorted(
                sources.values(), key=lambda x: x["name"]
            )
    except Exception:
        st.session_state.indexed_files = []


with tab_index:
    chroma_dir = st.session_state.chroma_dir
    md_out = st.session_state.markdown_out_dir

    col_idx_btn, col_idx_status = st.columns([2, 5])
    with col_idx_btn:
        build_btn = st.button(
            "🔧 构建 / 更新索引", type="primary", use_container_width=True
        )
    with col_idx_status:
        st.caption(f"源目录：`{md_out}` → 索引：`{chroma_dir}`")

    if build_btn:
        embed_key = st.session_state.get("embed_api_key", "")
        if not embed_key:
            st.error("⚠️ 请先在『RAG 配置』中填写 Embedding API Key。")
        else:
            from ragbot import build_vectorstore

            with st.status("正在构建向量索引...", expanded=True) as build_status:

                def idx_progress(cur, total, msg):
                    st.write(f"{msg}")

                try:
                    build_vectorstore(
                        md_dir=md_out,
                        embed_api_key=embed_key,
                        embed_base_url=st.session_state.get("embed_base_url", ""),
                        embed_model=st.session_state.get("embed_model", ""),
                        persist_dir=chroma_dir,
                        progress_callback=idx_progress,
                    )
                    build_status.update(
                        label="✅ 索引构建完成！", state="complete"
                    )
                    st.session_state.vectorstore_ready = True
                    # 刷新索引文件清单
                    _load_manifest(chroma_dir)
                except Exception as exc:
                    build_status.update(
                        label=f"❌ 构建失败: {exc}", state="error"
                    )

    # ── 已索引文件列表 ──
    st.markdown("---")
    st.markdown("**📋 当前索引包含的文件**")
    _load_manifest(chroma_dir)

    indexed_files = st.session_state.get("indexed_files", [])
    build_time = st.session_state.get("index_build_time", "")

    if not indexed_files:
        st.caption("暂无索引文件。请先导出聊天记录，再构建索引。")
    else:
        if build_time:
            st.caption(f"🕐 索引构建时间：{build_time}")

        import pandas as pd

        df = pd.DataFrame(indexed_files)
        column_config = {"name": "文件名"}
        if "size_kb" in df.columns:
            column_config["size_kb"] = st.column_config.NumberColumn(
                "大小 (KB)", format="%.1f"
            )
        if "mtime" in df.columns:
            column_config["mtime"] = "修改时间"
        if "chunks" in df.columns:
            column_config["chunks"] = st.column_config.NumberColumn("分片数")

        st.dataframe(
            df,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
        )

        total_chunks = sum(f.get("chunks", 0) for f in indexed_files)
        st.caption(f"共 {len(indexed_files)} 个文件，{total_chunks} 个对话片段")
