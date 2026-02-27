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

.hero-title {
    background: linear-gradient(135deg, #4ade80 0%, #22d3ee 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
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
</style>
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

st.markdown(
    """
<div class="brand-card">
    <div class="hero-title">💬 OpenCortex</div>
    <div class="hero-sub">本地知识库问答工具</div>
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
