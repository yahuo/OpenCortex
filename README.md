# OpenCortex

精简通用版，本地目录自动建索引 + 对话问答。

启动应用时先自动重建向量索引；终端输出构建进度。索引完成后页面即可直接问答。

## 功能

- 单页面应用（无配置页）
- 读取本地目录并递归索引文本文件
- 启动阶段自动重建索引（进度在启动命令输出）
- 页面不再提供索引构建步骤
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
python3 start.py
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

# 可选：启动地址
APP_HOST=127.0.0.1
APP_PORT=8501
```

## 使用说明

1. 执行 `python3 start.py` 启动应用。
2. 索引重建阶段不会自动打开浏览器。
3. 终端出现“请访问: http://... ”后，手动打开该地址开始问答。
4. 在输入框提问，系统基于索引内容生成回答。

## 目录结构（当前核心）

- `start.py`：启动器（先重建索引，再启动 Web 服务）
- `app.py`：单页 UI 与对话流程
- `ragbot.py`：文档分片、向量构建、检索问答
- `.env.example`：环境变量模板
