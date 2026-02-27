import streamlit as st


def inject_global_css():
    """注入全局自定义 CSS 样式"""
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
