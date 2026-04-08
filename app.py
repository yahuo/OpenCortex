"""OpenCortex — 单页对话界面。"""
import html
import json
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from ragbot import ask_stream as rag_ask_stream
from ragbot import load_search_bundle
from wiki import save_query_note

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

.summary-card {
    background: linear-gradient(180deg, rgba(15,23,42,0.78), rgba(15,23,42,0.55));
    border: 1px solid rgba(148,163,184,0.16);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    min-height: 112px;
    margin-bottom: 0.7rem;
}
.summary-label {
    color: #94a3b8;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.35rem;
}
.summary-value {
    color: #e2e8f0;
    font-size: 1.7rem;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.summary-hint {
    color: #64748b;
    font-size: 0.84rem;
    line-height: 1.45;
}
.community-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    min-height: 170px;
}
.community-title {
    color: #e2e8f0;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.community-meta {
    color: #94a3b8;
    font-size: 0.82rem;
    line-height: 1.5;
}
.token-pill {
    display: inline-block;
    background: rgba(34,211,238,0.08);
    border: 1px solid rgba(34,211,238,0.15);
    color: #67e8f9;
    border-radius: 999px;
    padding: 0.14rem 0.55rem;
    font-size: 0.76rem;
    margin: 0 0.35rem 0.35rem 0;
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


@st.cache_data(show_spinner=False)
def load_structure_summary(
    persist_dir: str,
    index_mtime: float,
) -> dict[str, Any]:
    """读取离线结构产物。index_mtime 仅用于缓存失效。"""
    del index_mtime

    persist_path = Path(persist_dir)
    community_index_path = persist_path / "community_index.json"
    graph_report_path = persist_path / "reports" / "GRAPH_REPORT.md"
    lint_report_path = persist_path / "lint_report.json"

    community_index: dict[str, Any] | None = None
    lint_report: dict[str, Any] | None = None
    graph_report = ""

    if community_index_path.exists():
        try:
            payload = json.loads(community_index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            community_index = payload

    if graph_report_path.exists():
        try:
            graph_report = graph_report_path.read_text(encoding="utf-8").strip()
        except OSError:
            graph_report = ""

    if lint_report_path.exists():
        try:
            payload = json.loads(lint_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            lint_report = payload

    return {
        "community_index": community_index,
        "lint_report": lint_report,
        "graph_report": graph_report,
        "community_index_path": str(community_index_path),
        "lint_report_path": str(lint_report_path),
        "graph_report_path": str(graph_report_path),
    }


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


def render_query_note_actions(message_index: int, message: dict[str, Any], persist_dir: str) -> None:
    if message.get("role") != "assistant":
        return

    question = str(message.get("question", "") or "").strip()
    answer = str(message.get("content", "") or "").strip()
    sources = message.get("sources", [])
    if not question or not answer or not isinstance(sources, list) or not sources:
        return

    saved_relpath = str(message.get("query_note_relpath", "") or "").strip()
    save_error = str(message.get("query_note_error", "") or "").strip()
    button_key = f"save-query-note-{message_index}"

    left, right = st.columns([1, 4])
    with left:
        clicked = st.button("保存为知识", key=button_key, disabled=bool(saved_relpath))
    with right:
        if saved_relpath:
            st.caption(f"已保存到 `{saved_relpath}`")
        else:
            st.caption("显式接受后回写，默认不进入最高优先级检索证据链。")
        if save_error:
            st.error(f"保存失败：{save_error}")

    if not clicked:
        return

    try:
        result = save_query_note(
            persist_path=Path(persist_dir),
            question=question,
            answer=answer,
            sources=sources,
        )
    except Exception as exc:
        st.session_state.rag_messages[message_index]["query_note_error"] = str(exc)
        st.rerun()
        return

    st.session_state.rag_messages[message_index]["query_note_relpath"] = result["note_relpath"]
    st.session_state.rag_messages[message_index]["query_note_error"] = ""
    st.rerun()


def render_stat_card(label: str, value: str, hint: str) -> None:
    st.markdown(
        f"""<div class="summary-card">
    <div class="summary-label">{html.escape(label)}</div>
    <div class="summary-value">{html.escape(value)}</div>
    <div class="summary-hint">{html.escape(hint)}</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_structure_summary(summary: dict[str, Any]) -> None:
    community_index = summary.get("community_index")
    if not isinstance(community_index, dict):
        return

    communities = community_index.get("communities", [])
    god_nodes = community_index.get("god_nodes", [])
    bridges = community_index.get("bridges", [])
    lint_report = summary.get("lint_report")
    lint_summary = lint_report.get("summary", {}) if isinstance(lint_report, dict) else {}
    lint_issue_count = sum(
        int(lint_summary.get(key, 0) or 0)
        for key in ("stale_pages", "orphan_pages", "missing_links")
    )
    lint_report_path = str(summary.get("lint_report_path", "") or "")
    graph_report = str(summary.get("graph_report", "") or "")
    graph_report_path = str(summary.get("graph_report_path", "") or "")

    stats = [
        (
            "社区",
            str(community_index.get("community_count", len(communities))),
            "按强关系边聚合出的主题簇",
        ),
        (
            "文件",
            str(community_index.get("file_count", 0)),
            "参与结构图谱的离线文件总数",
        ),
        (
            "关键节点",
            str(len(god_nodes)),
            "连接多个上下文的高频枢纽",
        ),
        (
            "健康检查",
            str(lint_issue_count),
            "wiki / query note / 链接的待处理问题",
        ),
        (
            "跨社区桥",
            str(len(bridges)),
            "帮助从一个主题跳到另一个主题",
        ),
    ]

    stat_columns = st.columns(len(stats))
    for column, (label, value, hint) in zip(stat_columns, stats):
        with column:
            render_stat_card(label, value, hint)

    with st.expander("查看知识结构摘要", expanded=False):
        st.caption("这些摘要来自离线生成的 community index、lint report 和结构报告，不参与本轮检索排序。")
        tabs = st.tabs(["核心社区", "关键枢纽", "桥接关系", "健康检查", "结构报告"])

        with tabs[0]:
            top_communities = [item for item in communities if isinstance(item, dict)][:3]
            if not top_communities:
                st.info("当前还没有可展示的社区摘要。")
            else:
                community_columns = st.columns(len(top_communities))
                for column, community in zip(community_columns, top_communities):
                    label = str(community.get("label") or community.get("id") or "未命名社区")
                    size = int(community.get("size") or 0)
                    top_files = ", ".join(
                        str(item.get("source", ""))
                        for item in community.get("top_files", [])
                        if isinstance(item, dict) and item.get("source")
                    )
                    top_symbols = [
                        str(item.get("name", ""))
                        for item in community.get("top_symbols", [])
                        if isinstance(item, dict) and item.get("name")
                    ][:3]
                    suggested_queries = [
                        str(item)
                        for item in community.get("suggested_queries", [])
                        if isinstance(item, str) and item.strip()
                    ][:2]
                    with column:
                        st.markdown(
                            f"""<div class="community-card">
    <div class="community-title">{html.escape(label)}</div>
    <div class="community-meta">文件数：{size}</div>
    <div class="community-meta">重点文件：{html.escape(top_files or '暂无')}</div>
    <div class="community-meta">推荐问题：{html.escape('；'.join(suggested_queries) or '暂无')}</div>
</div>""",
                            unsafe_allow_html=True,
                        )
                        if top_symbols:
                            st.markdown(
                                "".join(
                                    f'<span class="token-pill">{html.escape(symbol)}</span>'
                                    for symbol in top_symbols
                                ),
                                unsafe_allow_html=True,
                            )

        with tabs[1]:
            if not god_nodes:
                st.info("当前没有识别到关键枢纽节点。")
            else:
                for node in god_nodes[:8]:
                    if not isinstance(node, dict):
                        continue
                    label = str(node.get("name") or node.get("source") or "未命名节点")
                    source = str(node.get("source") or "未知来源")
                    degree = int(node.get("degree") or 0)
                    st.markdown(f"- `{label}` 来自 `{source}`，连接度 `{degree}`")

        with tabs[2]:
            if not bridges:
                st.info("当前没有跨社区桥接关系。")
            else:
                for bridge in bridges[:8]:
                    if not isinstance(bridge, dict):
                        continue
                    source = str(bridge.get("source") or "未知来源")
                    target = str(bridge.get("target") or "未知目标")
                    kind = str(bridge.get("kind") or "bridge")
                    st.markdown(f"- `{source}` -> `{target}` · `{kind}`")

        with tabs[3]:
            if lint_issue_count == 0:
                st.success("当前没有检测到 stale page、orphan page 或 missing link。")
            else:
                st.warning(f"当前共有 {lint_issue_count} 个健康检查问题，建议先处理后再依赖 wiki 导航。")

            issue_specs = [
                ("stale_pages", "过期页面", "page"),
                ("orphan_pages", "孤儿页面", "page"),
                ("missing_links", "失效链接", "page"),
            ]
            for key, title, page_key in issue_specs:
                issues = lint_report.get(key, []) if isinstance(lint_report, dict) else []
                if not issues:
                    st.markdown(f"**{title}**：0")
                    continue
                st.markdown(f"**{title}**：{len(issues)}")
                for issue in issues[:8]:
                    if not isinstance(issue, dict):
                        continue
                    page = str(issue.get(page_key) or "unknown")
                    reason = str(issue.get("reason") or issue.get("target") or "")
                    suffix = f" · `{reason}`" if reason else ""
                    st.markdown(f"- `{page}`{suffix}")
            if lint_report_path:
                st.caption(f"健康检查路径：{lint_report_path}")

        with tabs[4]:
            if graph_report:
                st.caption(f"结构报告路径：{graph_report_path}")
                st.markdown(graph_report)
            else:
                st.info("当前没有可展示的结构报告。")


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

structure_summary = load_structure_summary(
    persist_dir=cfg["persist_dir"],
    index_mtime=current_mtime,
)

search_bundle = get_search_bundle(
    embed_api_key=cfg["embed_api_key"],
    embed_base_url=cfg["embed_base_url"],
    embed_model=cfg["embed_model"],
    persist_dir=cfg["persist_dir"],
)

if search_bundle is None:
    st.error("索引加载失败，请重启应用后重试。")
    st.stop()

render_structure_summary(structure_summary)

for msg_index, msg in enumerate(st.session_state.rag_messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])
        if msg.get("bridge_entities"):
            render_bridge_entities(msg["bridge_entities"])
        if msg.get("search_trace"):
            render_search_trace(msg["search_trace"])
        if msg.get("role") == "assistant":
            render_query_note_actions(msg_index, msg, cfg["persist_dir"])

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
            "question": question,
            "content": answer,
            "sources": sources,
            "bridge_entities": bridge_entities,
            "search_trace": search_trace,
            "query_note_relpath": "",
            "query_note_error": "",
        }
    )
