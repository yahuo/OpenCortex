# OpenCortex

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/向量库-FAISS-green.svg)](https://github.com/facebookresearch/faiss)

**轻量、本地优先的 RAG 对话工具。** 指定一个本地目录，自动建立向量索引，然后用任何 OpenAI 兼容的 LLM 与你的文档对话——数据全部留在你自己的机器上。

[English README](README.md)

---

## 功能亮点

- **零配置单页界面** — Streamlit，无需配置页
- **递归索引** — 支持 `.md`、`.txt`、`.json`、`.yaml`、`.csv`、`.rst`、`.log` 等
- **微信导出支持** — 原生解析微信导出的 Markdown，按时间窗口分片
- **LLM 和 Embedding 可插拔** — 任意 OpenAI 兼容接口：硅基流动、Gemini、DeepSeek、Kimi、GLM 等
- **本地 FAISS 向量库** — 无云端依赖，数据不离本机
- **索引热加载** — 更新文档后重建索引，无需重启服务，刷新页面即生效
- **Docker 部署** — 将同一知识库以局域网服务形式共享给多人

---

## 架构

```
本地文档目录
      │
      ▼ ragbot.py — 解析、分片、向量化 → FAISS 索引（本地持久化）
      │
      ▼ app.py — Streamlit 界面，加载 FAISS，处理对话
      │
      ▼ 任意 OpenAI 兼容 LLM（流式输出）
```

`start.py` 负责协调：重建索引 → 启动 Web 服务。

---

## 快速开始

### 本地运行（个人使用推荐）

需要 Python 3.12。

```bash
git clone https://github.com/yahuo/OpenCortex.git
cd WechatLLM

python3.12 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# 编辑 .env，至少设置 LOCAL_DOCS_DIR、EMBED_API_KEY、LLM_API_KEY
```

启动（自动重建索引后启动 Web 服务）：

```bash
source venv/bin/activate
python3 start.py
```

浏览器访问 `http://localhost:8501`。

### Docker 部署（局域网多用户）

需要 Docker 和 Docker Compose v2。

```bash
cp .env.example .env
# 编辑 .env，设置 LOCAL_DOCS_DIR、EMBED_API_KEY、LLM_API_KEY
# LOCAL_DOCS_DIR 和 CHROMA_PERSIST_DIR 建议使用绝对路径
```

**首次部署：**

```bash
docker compose build
docker compose run --rm app python start.py --rebuild-only   # 建索引
docker compose up -d                                          # 启动服务
```

通过 `http://<服务器IP>:8501` 访问。

**文档更新后热加载（无需重启容器）：**

```bash
docker compose exec app python start.py --rebuild-only
# 刷新浏览器后自动加载新索引
```

**常用运维命令：**

```bash
docker compose logs -f    # 查看日志
docker compose down       # 停止服务
docker compose up -d      # 重新启动
```

---

## 配置项

将 `.env.example` 复制为 `.env` 并填写所需参数。

| 变量 | 必填 | 默认值 | 说明 |
|---|:---:|---|---|
| `LOCAL_DOCS_DIR` | ✅ | `./docs` | 要索引的目录（递归） |
| `EMBED_API_KEY` | ✅ | — | Embedding 服务 API Key |
| `EMBED_BASE_URL` | | `https://api.siliconflow.cn/v1` | Embedding API 地址 |
| `EMBED_MODEL` | | `BAAI/bge-m3` | Embedding 模型名称 |
| `LLM_API_KEY` | ✅ | — | LLM 服务 API Key |
| `LLM_BASE_URL` | | `https://generativelanguage.googleapis.com/v1beta/openai/` | LLM API 地址 |
| `LLM_MODEL` | | `gemini-2.0-flash` | LLM 模型名称 |
| `CHROMA_PERSIST_DIR` | | `~/wechat_rag_db` | FAISS 索引持久化目录 |
| `APP_HOST` | | `127.0.0.1` | 服务绑定地址 |
| `APP_PORT` | | `8501` | 服务端口 |

**兼容的 LLM 服务商**（设置对应的 `LLM_BASE_URL`）：

| 服务商 | Base URL |
|---|---|
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| DeepSeek | `https://api.deepseek.com/v1` |
| Kimi（Moonshot） | `https://api.moonshot.cn/v1` |
| GLM（智谱） | `https://open.bigmodel.cn/api/paas/v4/` |
| 硅基流动 | `https://api.siliconflow.cn/v1` |

---

## 支持的文件类型

| 扩展名 | 分片策略 |
|---|---|
| `.md`、`.markdown`、`.mdx` | 微信导出格式 → 时间窗口分片；其他 Markdown → 固定大小分片 |
| `.txt`、`.rst`、`.log` | 固定大小分片（含重叠） |
| `.csv`、`.json`、`.yaml`、`.yml` | 固定大小分片（含重叠） |

---

## 目录结构

```
OpenCortex/
├── start.py          # 启动器：重建索引 → 启动 Web 服务
├── app.py            # Streamlit 单页界面
├── ragbot.py         # 核心：分片、向量化、FAISS、RAG 问答
├── Dockerfile        # 容器镜像定义
├── docker-compose.yml# 多用户部署编排
├── requirements.txt  # Python 依赖
└── .env.example      # 环境变量模板
```

---

## 参与贡献

欢迎提交 Issue 和 Pull Request。

1. Fork 本仓库
2. 新建分支（`git checkout -b feature/你的功能`）
3. 提交改动
4. 发起 Pull Request

---

## 许可证

[MIT](LICENSE) © 2024 OpenCortex contributors
