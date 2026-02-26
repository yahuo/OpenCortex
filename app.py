import streamlit as st
import pandas as pd
import sqlite3
import os
from export_group import export_by_username, CONTACT_DB, OUT_DIR

st.set_page_config(page_title="WechatLLM 微信知识库导出工具", page_icon="💬", layout="wide")

# ================================
# Core Logic
# ================================
@st.cache_data
def load_all_contacts():
    if not os.path.exists(CONTACT_DB):
        return pd.DataFrame()
        
    conn = sqlite3.connect(CONTACT_DB)
    # 取出联系人的核心信息，排查无备注名称的空对象
    df = pd.read_sql_query("SELECT username, nick_name, remark FROM contact", conn)
    conn.close()
    
    # 丰富前端显示逻辑
    df['Display Name'] = df['remark'].where(df['remark'] != '', df['nick_name'])
    df['Type'] = df['username'].apply(lambda x: '🌐 群聊 (Group)' if '@chatroom' in x else '👤 私聊 (Contact)')
    
    # 去除无名字段
    df = df[df['Display Name'].notna() & (df['Display Name'] != '')]
    
    return df[['Display Name', 'Type', 'username']].copy()

# ================================
# UI Rendering
# ================================
st.title("💬 WechatLLM 聊天记录导出大盘")
st.markdown("通过此控制台，你可以在本地无极搜索、预览并批量导出你想要分析或投喂给大模型的微信会话集合。")

df_contacts = load_all_contacts()

if df_contacts.empty:
    st.error("❌ 无法检测到可用的本地花名册。请确保你已经成功执行了 README.md 中的 `chatlog_bin decrypt` 脱壳挂载步骤。")
    st.stop()

st.subheader("📚 本地联络人列表 (搜索与多选)")

# 采用分栏设计：让搜索框和导出按钮同一行
col_search, col_btn = st.columns([5, 1])

with col_search:
    search_query = st.text_input("🔍 搜索联系人和群聊", placeholder="例如：技术部", label_visibility="collapsed").strip()

# 进度区位置预留
progress_container = st.empty()
status_text = st.empty()

# Apply search filter
if search_query:
    filtered_df = df_contacts[df_contacts['Display Name'].str.contains(search_query, case=False, na=False)].copy()
else:
    filtered_df = df_contacts.copy()

# 插入一列用于布尔值勾选框
if "Export" not in filtered_df.columns:
    filtered_df.insert(0, "Export", False)
else:
    filtered_df["Export"] = False

# 构建数据编辑网格
# 去掉强制的 height 参数，让其自适应内容；通过使用 None 作为高度，避免内部 Y 轴小滚动条。
edited_df = st.data_editor(
    filtered_df,
    column_config={
        "Export": st.column_config.CheckboxColumn("导出?", help="勾选后可以批量导出", default=False),
        "Display Name": "对话名称",
        "Type": "类型",
        "username": "内部 ID (md5)"
    },
    disabled=["Display Name", "Type", "username"],
    hide_index=True,
    use_container_width=True
)

# ================================
# Export Action (Rendered at top placeholder)
# ================================
selected_rows = edited_df[edited_df["Export"] == True]

with col_btn:
    # 这里按钮被放入顶部的列中，但是逻辑被放在下面获取 edited_df 之后，巧妙绕过上下层级
    btn_disabled = selected_rows.empty
    start_export = st.button("🚀 批量提取选定", type="primary", use_container_width=True, disabled=btn_disabled)

if not selected_rows.empty:
    st.success(f"已高亮选中 {len(selected_rows)} 个会话对象等待提出。")

if start_export and not selected_rows.empty:
    my_bar = progress_container.progress(0, text="准备开始...")
    status_text.info("正在执行批量解压...")
    
    success_count = 0
    
    # 针对每个被勾选项执行动作
    for i, row in selected_rows.iterrows():
        display_name = row['Display Name']
        username = row['username']
        
        st.toast(f"开始提取: {display_name}")
        
        # 使用包装回调的钩子实时捕获底层的导出进度并更新 Streamlit GUI
        def on_progress(current, total, msg):
            percent = current / total if total > 0 else 0
            my_bar.progress(percent, text=f"[{display_name}] {msg}")
        
        try:
            success, reason_or_file = export_by_username(username, display_name, progress_callback=on_progress)
            if success:
                success_count += 1
            else:
                st.error(f"提取 {display_name} 失败: {reason_or_file}")
        except Exception as e:
            st.error(f"严重崩溃跳过 {display_name}: {e}")
            
    my_bar.empty()
    if success_count == len(selected_rows):
        status_text.success(f"🎉 全部大功告成！成功导出了 {success_count} 个 Markdown 纯净文件到 {OUT_DIR}")
        st.balloons()
    else:
        status_text.warning(f"完成。成功: {success_count}，失败: {len(selected_rows) - success_count}。请检查上方报错原因。")
