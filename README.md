# OpenCortex

精简通用版，本地目录自动建索引 + 对话问答。

打开页面后会自动重建向量索引，构建完成后触发气球提示，然后即可提问。

## 功能

- 单页面应用（无配置页）
- 读取本地目录并递归索引文本文件
- 索引过程中显示进度条
- 索引完成后隐藏进度信息，仅显示完成气球动画
- 基于本地 FAISS 向量库进行 RAG 对话

## 支持文件类型

`ragbot.py` 当前支持以下后缀：

- `.md`
- `.markdown`
- `.mdx`
- `.txt`
- `.rst`
- `.log`
- `.csv`
- `.json`
- `.yaml`
- `.yml`

说明：
- 微信导出样式的 Markdown 会按时间窗口分片。
- 其他文本按固定窗口分片。

## 快速开始

```bash
git clone <repo_url>
cd OpenCortex

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，至少设置：

- `LOCAL_DOCS_DIR`
- `EMBED_API_KEY`
- `LLM_API_KEY`

启动：

```bash
source venv/bin/activate
streamlit run app.py
```

浏览器访问：`http://localhost:8501`

## 环境变量

```env
LOCAL_DOCS_DIR=./docs

EMBED_API_KEY=...
EMBED_BASE_URL=https://api.siliconflow.cn/v1
EMBED_MODEL=BAAI/bge-m3

LLM_API_KEY=...
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-2.0-flash

CHROMA_PERSIST_DIR=~/wechat_rag_db
```

## 使用说明

1. 打开页面，系统自动重建索引。
2. 构建完成后看到气球动画，表示可开始问答。
3. 在输入框提问，系统基于索引内容生成回答。
4. 可在侧边栏手动点击“重新创建索引”。

## 目录结构（当前核心）

- `app.py`：单页 UI 与自动索引流程
- `ragbot.py`：文档分片、向量构建、检索问答
- `.env.example`：环境变量模板
