# WechatLLM — Mac 微信本地知识库提取工具箱 🚀

从本地微信数据库中完整导出聊天记录，生成结构化 Markdown 文件，可直接用于 GPT / Claude 知识库微调或本地 RAG 问答。

**支持环境**：macOS 微信 v4.x（原生新款）｜ Python 3.11+

---

## 🖥️ 快速开始（可视化界面）

### 1. 环境准备

```bash
# 克隆项目并进入目录
git clone <repo_url> && cd WechatLLM

# 创建虚拟环境并安装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置 API Key（用于 RAG 问答）

```bash
cp .env.example .env
# 用编辑器打开 .env，填入以下内容：
#   EMBED_API_KEY    — Embedding 服务的 API Key
#   EMBED_BASE_URL   — Embedding 服务地址（默认硅基流动，可换其他兼容 OpenAI 的服务）
#   EMBED_MODEL      — Embedding 模型名称（默认 BAAI/bge-m3）
#   LLM_API_KEY      — LLM 的 API Key（Gemini / Kimi / GLM 等）
#   LLM_BASE_URL     — LLM 的 base_url
#   LLM_MODEL        — LLM 模型名称（默认 gemini-2.0-flash）
```

### 3. 启动可视化控制台

```bash
source venv/bin/activate
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`，页面功能说明：

| 左侧边栏 | 右侧主区域 |
|---|---|
| 🔓 **解密数据库**：填密钥 + 账号目录，一键解密 | 📋 **联系人列表**：自动加载，支持搜索 |
| 📁 **导出目录**：配置 Markdown 输出路径 | ☑️ 勾选会话后批量导出，带进度条 |
| 🧠 **RAG 问答配置**：填入 Embedding 和 LLM Key | 💬 **知识库问答**：语义检索 + 带引用回答 |

> **已有解密数据？** 若 `~/wechat_db_backup/db_storage` 下已有数据，启动后联系人列表**自动展示**，无需重复解密。

---

## 🔑 步骤 0：获取微信数据库密钥

> [!WARNING]
> **必须提前关闭 macOS SIP（系统完整性保护）**
>
> 运行 `csrutil status` 确认是否已关闭。若未关闭：
> 1. 完全关机 → 长按电源键进入恢复模式（Intel Mac：`Command+R`）
> 2. 打开终端，执行 `csrutil disable` → 重启
> 3. *(导出完成后可重新执行 `csrutil enable` 恢复保护)*

**密钥提取步骤：**

1. 彻底退出微信（状态栏也不能有微信图标）
2. 打开终端，执行：
   ```bash
   sudo lldb -n WeChat -w
   ```
3. **立刻点击打开微信**，等待终端出现 `(lldb)` 提示符
4. 执行以下命令打断点并继续：
   ```text
   br set -n CCKeyDerivationPBKDF
   c
   ```
5. **在手机上扫码登录微信**，微信会再次被断点捕获
6. 在 `(lldb)` 中读取密钥：
   ```text
   memory read --size 1 --format x --count 32 $x1
   ```
   将输出的 32 个 `0x??` 去掉 `0x` 拼接成 64 位 hex 字符串，即为密钥
7. 执行 `detach` → `quit` 释放微信

将密钥保存至 `wechat_db_key.txt` 备用。

---

## 🔓 解密数据库

在可视化界面左侧"🔓 解密数据库"区域填入：
- **数据库密钥**（64 位 hex）
- **微信账号目录**（自动识别，下拉选择）
- **解密输出目录**（默认 `~/wechat_db_backup`）

点击"🔑 开始解密"即可。解密完成后右侧联系人列表自动刷新。

---

## 🧠 知识库问答（RAG）

1. 先在右侧勾选会话 → 点"🚀 导出选定"导出为 Markdown
2. 展开左侧"🧠 RAG 问答配置"，确认 Key 已填入（从 `.env` 自动读取）
3. 右侧底部点"🔧 构建 / 更新索引"（使用 FAISS + bge-m3 向量化）
4. 索引完成后直接在聊天框提问，回答附带原文引用时间段

**支持切换 LLM 模型**（仅改 `.env` 三行即可）：

| 模型 | LLM_BASE_URL |
|---|---|
| Gemini（默认）| `https://generativelanguage.googleapis.com/v1beta/openai/` |
| Kimi | `https://api.moonshot.cn/v1` |
| GLM | `https://open.bigmodel.cn/api/paas/v4/` |
| DeepSeek | `https://api.deepseek.com/v1` |

---

## ⌨️ 极客模式（仅命令行）

如果不想启动 Web UI，可直接调用 Python 脚本：

**解密数据库：**
```bash
source venv/bin/activate
python3 decrypt_wechat_db.py decrypt \
  -k "你的64位密钥" \
  -p darwin -v 4 \
  -d "/Users/用户名/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/<账号目录>" \
  -o ~/wechat_db_backup
```

**导出指定会话：**
```bash
python3 export_group.py "公共技术部"
```

**可选环境变量：**
```bash
export WECHATLLM_DB_DIR=~/wechat_db_backup/db_storage
export WECHATLLM_OUT_DIR=~/wechat_export
```

---

## 🧯 常见问题

| 问题 | 排查方法 |
|---|---|
| 未检测到联系人库 | 确认 `~/wechat_db_backup/db_storage/contact/contact.db` 存在 |
| `查无数据库表: Msg_<md5>` | 该会话无本地消息或已清理，可跳过 |
| 图片提取失败但文本正常 | 部分图片为 V2 加密格式，不影响文本导出 |
| ChromaDB 报 Pydantic 错误 | 已替换为 FAISS，重装依赖 `pip install -r requirements.txt` |

---

尽情享用属于你自己的本地知识库进行 AI 对话与微调！🤖✨
