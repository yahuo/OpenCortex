"""OpenCortex — 单页对话界面。"""
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ragbot import ask_stream as rag_ask_stream
from ragbot import load_vectorstore

load_dotenv()

st.set_page_config(page_title="OpenCortex", page_icon="💬", layout="wide")


def inject_global_css() -> None:
    """注入接近 master 分支的全局样式，并隐藏顶部工具栏"""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* 共享渐变文字样式 */
.hero-title,
.empty-hero-title {
    background: linear-gradient(135deg, #4ade80 0%, #22d3ee 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
    line-height: 1.2;
}
.hero-title {
    font-size: 3rem;
    margin-bottom: 0.35rem;
}
.hero-sub { color: #64748b; font-size: 0.95rem; margin-bottom: 0; }

.brand-card {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
    margin: 0 0 1.2rem;
}

.block-container { padding-top: 1.6rem; padding-bottom: 2rem; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stToolbar"] {display: none !important;}
[data-testid="stDeployButton"] {display: none !important;}
[data-testid="stStatusWidget"] {display: none !important;}
[data-testid="stDecoration"] {display: none !important;}
[data-testid="stSidebar"] {display: none !important;}
[data-testid="stSidebarNav"] {display: none !important;}
[data-testid="collapsedControl"] {display: none !important;}

/* ── 空状态居中容器 ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 60vh;
    text-align: center;
    gap: 2rem;
}
.empty-hero-title {
    font-size: 2.8rem;
    margin: 0;
}
.empty-hero-sub {
    color: #64748b;
    font-size: 1rem;
    margin-top: -1rem;
}

/* ── 能力卡片 ── */
.capability-cards {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
    width: 100%;
    max-width: 680px;
}
.capability-card {
    flex: 1;
    min-width: 180px;
    max-width: 210px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem 1rem;
    transition: transform 0.18s ease, border-color 0.18s ease;
    cursor: default;
}
.capability-card:hover {
    transform: translateY(-3px);
    border-color: rgba(74,222,128,0.35);
}
.capability-card .cap-icon { font-size: 1.6rem; margin-bottom: 0.5rem; }
.capability-card .cap-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.3rem;
}
.capability-card .cap-desc {
    font-size: 0.78rem;
    color: #64748b;
    line-height: 1.5;
}

/* ── 输入框 focus 光效 ── */
[data-testid="stChatInputContainer"] {
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInputContainer"]:focus-within {
    outline: none;
    box-shadow: 0 0 0 2px rgba(74,222,128,0.4), 0 0 18px rgba(34,211,238,0.15);
    border-color: rgba(74,222,128,0.5) !important; /* 覆盖 Streamlit 框架注入的 inline border-color */
}

/* ── 模型信息小字 ── */
.model-hint {
    text-align: center;
    font-size: 0.72rem;
    color: #64748b;
    margin-bottom: 0.4rem;
    letter-spacing: 0.02em;
}

/* ── 用户消息气泡右对齐 ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:last-child {
    background: rgba(74,222,128,0.08);
    border: 1px solid rgba(74,222,128,0.15);
    border-radius: 14px;
    padding: 0.6rem 1rem;
    max-width: 80%;
}

/* ── 引用来源卡片 ── */
.source-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
.source-card .src-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.3rem;
}
.source-badge {
    background: rgba(74,222,128,0.12);
    color: #4ade80;
    border-radius: 5px;
    padding: 0.1rem 0.45rem;
    font-size: 0.75rem;
    font-weight: 500;
    font-family: monospace;
}
.source-time {
    color: #475569;
    font-size: 0.75rem;
}
.source-snippet {
    color: #94a3b8;
    font-size: 0.8rem;
    line-height: 1.55;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    """渲染无消息时的居中欢迎区与能力卡片。"""
    st.markdown(
        """
<div class="empty-state">
    <div>
        <div class="empty-hero-title">💬 OpenCortex</div>
        <div class="empty-hero-sub">连接你的知识库，直接提问</div>
    </div>
    <div class="capability-cards">
        <div class="capability-card">
            <div class="cap-icon">📝</div>
            <div class="cap-title">内容摘要</div>
            <div class="cap-desc">总结某个话题或时间段的内容</div>
        </div>
        <div class="capability-card">
            <div class="cap-icon">🔍</div>
            <div class="cap-title">精准检索</div>
            <div class="cap-desc">描述你想找的信息，定位原文</div>
        </div>
        <div class="capability-card">
            <div class="cap-icon">💡</div>
            <div class="cap-title">深度解读</div>
            <div class="cap-desc">针对文档内容提问，获取详细分析</div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_vectorstore(
    embed_api_key: str,
    embed_base_url: str,
    embed_model: str,
    persist_dir: str,
):
    return load_vectorstore(
        embed_api_key=embed_api_key,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        persist_dir=persist_dir,
    )


def read_runtime_config() -> dict[str, str]:
    return {
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
    }


def init_session_state() -> None:
    st.session_state.setdefault("rag_messages", [])


init_session_state()
cfg = read_runtime_config()
inject_global_css()

if st.session_state.rag_messages:
    st.markdown(
        """
<div class="brand-card">
    <div class="hero-title">💬 OpenCortex</div>
    <div class="hero-sub">本地知识库问答工具</div>
</div>
""",
        unsafe_allow_html=True,
    )
else:
    render_empty_state()

missing_env = []
if not cfg["embed_api_key"]:
    missing_env.append("EMBED_API_KEY")
if not cfg["llm_api_key"]:
    missing_env.append("LLM_API_KEY")

if missing_env:
    st.error(f"缺少环境变量：{', '.join(missing_env)}")
    st.info("请在 .env 中配置后重新打开页面。")
    st.stop()

index_file = Path(cfg["persist_dir"]) / "index.faiss"
if not index_file.exists():
    st.error("未检测到向量索引。")
    st.code("python3 start.py", language="bash")
    st.info("请在终端运行上面的命令：先重建索引，再启动页面。")
    st.stop()

vectorstore = get_vectorstore(
    embed_api_key=cfg["embed_api_key"],
    embed_base_url=cfg["embed_base_url"],
    embed_model=cfg["embed_model"],
    persist_dir=cfg["persist_dir"],
)

if vectorstore is None:
    st.error("索引加载失败，请重启应用后重试。")
    st.stop()

for msg in st.session_state.rag_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("查看引用"):
                for src in msg["sources"]:
                    st.markdown(f"- `{src['source']}` {src['time_range']}")
                    st.caption(src["snippet"])

if question := st.chat_input("输入问题..."):
    st.session_state.rag_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.info("🔎 正在检索相关内容...")
        try:
            result = rag_ask_stream(
                question=question,
                vectorstore=vectorstore,
                llm_api_key=cfg["llm_api_key"],
                llm_model=cfg["llm_model"],
                llm_base_url=cfg["llm_base_url"],
            )
            sources = result["sources"]
            base_stream = result["answer_stream"]

            def stream_with_status():
                first_chunk = True
                for chunk in base_stream:
                    if first_chunk:
                        status_placeholder.empty()
                        first_chunk = False
                    yield chunk

            answer = st.write_stream(stream_with_status())
            status_placeholder.empty()
        except Exception as exc:
            answer = f"问答失败：{exc}"
            sources = []
            status_placeholder.empty()
            st.markdown(answer)

        if sources:
            with st.expander("查看引用"):
                for src in sources:
                    st.markdown(f"- `{src['source']}` {src['time_range']}")
                    st.caption(src["snippet"])

    st.session_state.rag_messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
