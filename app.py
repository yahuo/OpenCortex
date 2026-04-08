"""OpenCortex — 单页对话界面。"""
import html
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ragbot import ask_stream as rag_ask_stream
from ragbot import load_search_bundle

load_dotenv()

st.set_page_config(page_title="OpenCortex", page_icon="💬", layout="wide")


def inject_global_css() -> None:
    """注入接近 master 分支的全局样式，并隐藏顶部工具栏"""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* 渐变文字样式 */
.hero-title {
    display: inline-block;
    background: linear-gradient(90deg, #4ade80 0%, #22d3ee 50%, #818cf8 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1.2;
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



def render_sources(sources: list) -> None:
    """渲染引用来源卡片列表。"""
    with st.expander("查看引用"):
        for src in sources:
            location = src.get("time_range") or ""
            line_start = src.get("line_start")
            line_end = src.get("line_end")
            if not location and line_start:
                location = f"L{line_start}" if not line_end or line_end == line_start else f"L{line_start}-L{line_end}"
            badge = src.get("match_kind", "source")
            st.markdown(
                f"""<div class="source-card">
    <div class="src-meta">
        <span class="source-badge">{html.escape(src['source'])}</span>
        <span class="source-badge">{html.escape(badge)}</span>
        <span class="source-time">{html.escape(location)}</span>
    </div>
    <div class="source-snippet">{html.escape(src['snippet'])}</div>
</div>""",
                unsafe_allow_html=True,
            )


@st.cache_resource(show_spinner=False)
def get_search_bundle(
    embed_api_key: str,
    embed_base_url: str,
    embed_model: str,
    persist_dir: str,
):
    return load_search_bundle(
        embed_api_key=embed_api_key,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        persist_dir=persist_dir,
    )


@st.cache_resource
def _get_mtime_tracker() -> dict:
    """进程级单例，跨所有会话共享索引 mtime 追踪状态"""
    return {"last_mtime": 0.0}


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
        "search_mode": os.getenv("SEARCH_MODE", "hybrid").strip().lower() or "hybrid",
        "search_debug": os.getenv("SEARCH_DEBUG", "").strip().lower(),
    }


def init_session_state() -> None:
    st.session_state.setdefault("rag_messages", [])


def render_search_trace(search_trace: list[dict]) -> None:
    with st.expander("查看检索轨迹"):
        for step in search_trace:
            st.markdown(f"**{step.get('step', 'step')}**")
            st.json(step, expanded=False)


def render_bridge_entities(bridge_entities: list[dict]) -> None:
    with st.expander("查看桥接实体"):
        st.json(bridge_entities, expanded=False)


init_session_state()
cfg = read_runtime_config()
inject_global_css()

st.markdown(
    f"""
<div class="brand-card">
    <div class="hero-title">💬 OpenCortex</div>
    <div class="hero-sub">本地知识库问答 · {html.escape(cfg["llm_model"])}</div>
</div>
""",
    unsafe_allow_html=True,
)

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
    if Path("/.dockerenv").exists():
        st.code(
            "docker compose run --rm app python start.py --rebuild-only",
            language="bash",
        )
    else:
        st.code("python3 start.py", language="bash")
    st.info("请在终端运行上面的命令：先重建索引，再启动页面。")
    st.stop()

# 热加载：索引文件 mtime 变化时清除全局缓存（进程级哨兵，跨所有会话生效）
try:
    current_mtime = index_file.stat().st_mtime
except FileNotFoundError:
    st.error("索引文件已被移除，请重新运行 start.py 重建索引。")
    st.stop()
_mtime_tracker = _get_mtime_tracker()
if current_mtime != _mtime_tracker["last_mtime"]:
    get_search_bundle.clear()
    _mtime_tracker["last_mtime"] = current_mtime

search_bundle = get_search_bundle(
    embed_api_key=cfg["embed_api_key"],
    embed_base_url=cfg["embed_base_url"],
    embed_model=cfg["embed_model"],
    persist_dir=cfg["persist_dir"],
)

if search_bundle is None:
    st.error("索引加载失败，请重启应用后重试。")
    st.stop()

for msg in st.session_state.rag_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])
        if msg.get("bridge_entities"):
            render_bridge_entities(msg["bridge_entities"])
        if msg.get("search_trace"):
            render_search_trace(msg["search_trace"])

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
                search_bundle=search_bundle,
                llm_api_key=cfg["llm_api_key"],
                llm_model=cfg["llm_model"],
                llm_base_url=cfg["llm_base_url"],
                search_mode=cfg["search_mode"],
                debug=cfg["search_debug"] in {"1", "true", "yes", "on"},
            )
            sources = result["sources"]
            base_stream = result["answer_stream"]
            bridge_entities = result.get("bridge_entities", [])
            search_trace = result.get("search_trace", [])

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
            bridge_entities = []
            search_trace = []
            status_placeholder.empty()
            st.markdown(answer)

        if sources:
            render_sources(sources)
        if bridge_entities:
            render_bridge_entities(bridge_entities)
        if search_trace:
            render_search_trace(search_trace)

    st.session_state.rag_messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "bridge_entities": bridge_entities,
            "search_trace": search_trace,
        }
    )
