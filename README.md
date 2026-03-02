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

### 本地运行

推荐 Python 3.12。

```bash
git clone <repo_url>
cd OpenCortex

python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，至少设置：

- `LOCAL_DOCS_DIR`
- `EMBED_API_KEY`
- `LLM_API_KEY`

启动（自动重建索引后打开页面）：

```bash
source venv/bin/activate
python3 start.py
```

浏览器访问：`http://localhost:8501`

### Docker 部署（内网多用户）

适合将服务部署到局域网服务器，多人共享同一知识库。

**前置条件：** 安装 Docker 和 Docker Compose v2。

```bash
cp .env.example .env
# 编辑 .env，设置 LOCAL_DOCS_DIR、EMBED_API_KEY、LLM_API_KEY 等
```

**首次部署：**

```bash
docker compose build
docker compose run --rm app python start.py --rebuild-only  # 建索引
docker compose up -d                                         # 启服务
```

浏览器访问：`http://<服务器IP>:8501`

**文档更新后热加载（无需重启容器）：**

```bash
docker compose exec app python start.py --rebuild-only
```

页面刷新后自动加载新索引。

**常用运维命令：**

```bash
docker compose logs -f    # 查看日志
docker compose down       # 停止服务
docker compose up -d      # 重新启动
```

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
3. 终端出现”请访问: http://... “后，手动打开该地址开始问答。
4. 在输入框提问，系统基于索引内容生成回答。

## 目录结构（当前核心）

- `start.py`：启动器（先重建索引，再启动 Web 服务；`--rebuild-only` 仅建索引）
- `app.py`：单页 UI 与对话流程
- `ragbot.py`：文档分片、向量构建、检索问答
- `Dockerfile`：容器镜像定义
- `docker-compose.yml`：多用户部署编排
- `.env.example`：环境变量模板
