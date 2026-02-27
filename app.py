"""WechatLLM — Streamlit 多页面应用入口"""
import streamlit as st
from dotenv import load_dotenv

# 必须在导入其他模块之前加载环境变量
load_dotenv()

from shared.styles import inject_global_css
from shared.state import init_defaults

st.set_page_config(
    page_title="WechatLLM 微信知识库",
    page_icon="💬",
    layout="wide",
)

inject_global_css()
init_defaults()

# ── 页面路由 ──
chat_page = st.Page("pages/chat.py", title="对话", icon="💬", default=True)
settings_page = st.Page("pages/settings.py", title="配置", icon="⚙️")

pg = st.navigation([chat_page, settings_page])

# ── 侧边栏品牌 ──
with st.sidebar:
    st.markdown(
        '<div class="hero-title">💬 WechatLLM</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero-sub">微信知识库问答工具</div>', unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

pg.run()
