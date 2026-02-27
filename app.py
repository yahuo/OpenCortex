"""OpenCortex — 单页自动索引 + 对话"""
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ragbot import ask_stream as rag_ask_stream
from ragbot import build_vectorstore, load_vectorstore

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
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 0.15rem;
}
.hero-sub { color: #64748b; font-size: 0.85rem; margin-bottom: 0; }

.block-container { padding-top: 1.6rem; padding-bottom: 2rem; }

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

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stToolbar"] {display: none !important;}
[data-testid="stDeployButton"] {display: none !important;}
[data-testid="stStatusWidget"] {display: none !important;}
[data-testid="stDecoration"] {display: none !important;}
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
    source_dir = os.getenv("LOCAL_DOCS_DIR", "").strip() or "./docs"
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
    }


def init_session_state() -> None:
    st.session_state.setdefault("rag_messages", [])
    st.session_state.setdefault("index_ready", False)
    st.session_state.setdefault("index_error", "")


def auto_rebuild_index(cfg: dict[str, str]) -> bool:
    progress = st.progress(0.0, text="正在准备索引任务...")
    status = st.empty()

    def on_progress(current: int, total: int, message: str) -> None:
        ratio = 0.0 if total <= 0 else min(current / total, 1.0)
        progress.progress(ratio, text=message)
        status.caption(message)

    try:
        build_vectorstore(
            md_dir=cfg["source_dir"],
            embed_api_key=cfg["embed_api_key"],
            embed_base_url=cfg["embed_base_url"],
            embed_model=cfg["embed_model"],
            persist_dir=cfg["persist_dir"],
            progress_callback=on_progress,
        )
    except Exception as exc:
        st.session_state.index_ready = False
        st.session_state.index_error = str(exc)
        progress.empty()
        status.empty()
        st.error(f"自动创建向量索引失败：{exc}")
        return False

    get_vectorstore.clear()
    st.session_state.index_ready = True
    st.session_state.index_error = ""
    progress.empty()
    status.empty()
    st.balloons()
    return True


init_session_state()
cfg = read_runtime_config()
inject_global_css()

st.title("💬 OpenCortex")

with st.sidebar:
    st.markdown('<div class="hero-title">💬 OpenCortex</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">本地知识库问答工具</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 运行信息")
    st.code(
        (
            f"源目录: {cfg['source_dir']}\n"
            f"索引目录: {cfg['persist_dir']}\n"
            f"Embedding: {cfg['embed_model']}\n"
            f"LLM: {cfg['llm_model']}"
        ),
        language="text",
    )
    if st.button("🔁 重新创建索引", use_container_width=True):
        st.session_state.index_ready = False
        st.rerun()
    if st.button("🧹 清空对话", use_container_width=True):
        st.session_state.rag_messages = []
        st.rerun()

missing_env = []
if not cfg["embed_api_key"]:
    missing_env.append("EMBED_API_KEY")
if not cfg["llm_api_key"]:
    missing_env.append("LLM_API_KEY")

if missing_env:
    st.error(f"缺少环境变量：{', '.join(missing_env)}")
    st.info("请在 .env 中配置后重新打开页面。")
    st.stop()

source_path = Path(cfg["source_dir"])
if not source_path.is_dir():
    st.error(f"源目录不存在：{source_path}")
    st.info("请设置 .env 中的 LOCAL_DOCS_DIR。")
    st.stop()

index_notice = st.empty()
if not st.session_state.index_ready:
    index_notice.info("正在根据目录文件自动重建向量索引，请稍候...")
    if not auto_rebuild_index(cfg):
        st.stop()
    index_notice.empty()

vectorstore = get_vectorstore(
    embed_api_key=cfg["embed_api_key"],
    embed_base_url=cfg["embed_base_url"],
    embed_model=cfg["embed_model"],
    persist_dir=cfg["persist_dir"],
)

if vectorstore is None:
    st.error("索引加载失败，请点击侧边栏“重新创建索引”。")
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
        try:
            result = rag_ask_stream(
                question=question,
                vectorstore=vectorstore,
                llm_api_key=cfg["llm_api_key"],
                llm_model=cfg["llm_model"],
                llm_base_url=cfg["llm_base_url"],
            )
            sources = result["sources"]
            answer = st.write_stream(result["answer_stream"])
        except Exception as exc:
            answer = f"问答失败：{exc}"
            sources = []
            st.markdown(answer)

        if sources:
            with st.expander("查看引用"):
                for src in sources:
                    st.markdown(f"- `{src['source']}` {src['time_range']}")
                    st.caption(src["snippet"])

    st.session_state.rag_messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
