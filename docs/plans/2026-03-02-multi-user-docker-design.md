# 多用户 Docker 部署设计文档

日期：2026-03-02
范围：Docker Compose 容器化部署，支持内网多用户并发访问

## 目标

将当前本地单用户 Streamlit 应用改造为可在内网服务器部署的多用户服务，支持文档更新后热加载索引，无需重启容器。

## 关键结论

Streamlit 的 `st.session_state` 天然按浏览器 Tab 隔离，FAISS 只读查询线程安全，`@st.cache_resource` 已适合多用户共享同一份索引。**会话隔离无需任何代码修改。**

## 架构

```
┌─────────────────────────────────────────┐
│  docker-compose.yml                     │
│  ┌──────────────────────────────────┐   │
│  │  app (streamlit run app.py)      │   │
│  │  0.0.0.0:8501                    │   │
│  │  restart: unless-stopped         │   │
│  └──────────────────────────────────┘   │
│                                         │
│  volumes:                               │
│    ./wechat_export → /data/docs   (只读) │
│    ~/wechat_rag_db → /data/index  (读写) │
│    ./.env          → /app/.env    (只读) │
└─────────────────────────────────────────┘
```

## 工作流

| 场景 | 命令 |
|------|------|
| 首次部署 | `docker compose build` → `docker compose run --rm app python start.py --rebuild-only` → `docker compose up -d` |
| 日常启动 | `docker compose up -d` |
| 文档更新热加载 | `docker compose exec app python start.py --rebuild-only` |
| 停止服务 | `docker compose down` |

## 详细设计

### 1. 新增文件

**`Dockerfile`**
- 基础镜像：`python:3.12-slim`
- 安装 `requirements.txt` 依赖
- WORKDIR `/app`，COPY 项目文件
- EXPOSE 8501
- CMD：`python -m streamlit run app.py --server.address=0.0.0.0 --server.headless=true`

**`docker-compose.yml`**
- 单服务 `app`，`build: .`
- 端口：`${APP_PORT:-8501}:8501`
- volumes：`.env`（只读）、文档目录（只读）、索引目录（读写）
- `env_file: .env` + 覆盖 `LOCAL_DOCS_DIR=/data/docs`、`CHROMA_PERSIST_DIR=/data/index`
- `restart: unless-stopped`

**`.dockerignore`**
- 排除 `venv/`、`.env`、`wechat_export/`、`.worktrees/`、`__pycache__/`、`.git/`

### 2. 修改 `start.py`

加 `--rebuild-only` CLI 参数（`argparse`）：
- 有此参数：只执行 `rebuild_index()`，完成后退出 0
- 无此参数：现有行为不变（rebuild + launch streamlit）

### 3. 修改 `app.py`

**热加载机制：将索引文件修改时间作为 `get_vectorstore` 缓存 key 的一部分**

```python
@st.cache_resource(show_spinner=False)
def get_vectorstore(
    embed_api_key: str,
    embed_base_url: str,
    embed_model: str,
    persist_dir: str,
    _index_mtime: float,   # 下划线前缀 = Streamlit 不参与 hash，但参数变化会触发重载
):
    return load_vectorstore(...)
```

调用处：
```python
index_mtime = index_file.stat().st_mtime
vectorstore = get_vectorstore(..., _index_mtime=index_mtime)
```

当 `start.py --rebuild-only` 写入新索引后，`index.faiss` 的 mtime 变化，下一个页面请求自动触发缓存失效、重新加载向量库。

### 4. 更新 `.env.example`

添加：
```
APP_HOST=0.0.0.0
APP_PORT=8501
```

## 约束

- 不修改 `ragbot.py`
- 不引入新 Python 依赖
- 内网免登录，无需认证层
- 不支持 GPU（当前所有重计算在外部 API，本地 FAISS CPU 已足够）
