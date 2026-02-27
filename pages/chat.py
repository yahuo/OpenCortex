"""对话页面 — 纯 RAG 知识库问答界面"""
import json
from pathlib import Path

import streamlit as st

# ── 读取配置 ──
embed_api_key = st.session_state.get("embed_api_key", "")
llm_api_key = st.session_state.get("llm_api_key", "")
chroma_dir = st.session_state.get("chroma_dir", "")

# ── 前置检查 ──
if not embed_api_key or not llm_api_key:
    st.markdown("## 💬 知识库对话")
    st.info("🔑 请先前往『配置』页面填写 Embedding Key 和 LLM Key。")
    st.stop()

# ── 加载向量库 ──
from ragbot import load_vectorstore, ask as rag_ask

vectorstore = None
if st.session_state.get("vectorstore_ready") or (
    chroma_dir and Path(chroma_dir).exists()
):
    try:
        vectorstore = load_vectorstore(
            embed_api_key=embed_api_key,
            embed_base_url=st.session_state.get("embed_base_url", ""),
            embed_model=st.session_state.get("embed_model", ""),
            persist_dir=chroma_dir,
        )
        if vectorstore:
            st.session_state.vectorstore_ready = True
    except Exception:
        pass

if vectorstore is None:
    st.markdown("## 💬 知识库对话")
    st.info("📦 未检测到向量索引，请先前往『配置』页面导出聊天记录并构建索引。")
    st.stop()

# ── 侧边栏：轻量索引状态 ──
with st.sidebar:
    # 尝试加载 manifest 信息
    manifest_path = Path(chroma_dir) / "index_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = manifest.get("files", [])
            build_time = manifest.get("build_time", "")
            total_chunks = manifest.get("total_chunks", 0)
            st.caption(f"📦 已索引 {len(files)} 个文件 / {total_chunks} 个片段")
            if build_time:
                st.caption(f"🕐 {build_time}")
        except Exception:
            pass

# ── 聊天界面 ──
st.markdown("## 💬 知识库对话")

for msg in st.session_state.rag_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 查看引用来源"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**📄 {src['source']}** `{src['time_range']}`"
                    )
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
                    llm_model=st.session_state.get("llm_model", ""),
                    llm_base_url=st.session_state.get("llm_base_url", ""),
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
                    st.markdown(
                        f"**📄 {src['source']}** `{src['time_range']}`"
                    )
                    st.caption(src["snippet"])

    st.session_state.rag_messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
