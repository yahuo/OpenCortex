# Docker 多用户部署实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** 将 OpenCortex 改造为可 Docker Compose 部署的内网多用户服务，支持文档更新后热加载索引。

**Architecture:** 新增 `Dockerfile`/`docker-compose.yml`/`.dockerignore` 三个文件；修改 `start.py` 加 `--rebuild-only` 参数（只建索引不启 Streamlit）；修改 `app.py` 在索引文件 mtime 变化时自动清除 `@st.cache_resource` 缓存（热加载）。Streamlit 会话隔离天然存在，无需额外改动。

**Tech Stack:** Python 3.12, Streamlit, FAISS, Docker Compose v2, argparse

---

### Task 1：`start.py` 加 `--rebuild-only` 参数

**Files:**
- Modify: `start.py:1-10`（import 区域）
- Modify: `start.py:91-102`（`main()` 函数）

**Step 1: 在文件顶部加 argparse import**

在 `start.py` 第 8 行 `from pathlib import Path` 之后添加：

```python
import argparse
```

**Step 2: 修改 `main()` 函数，加参数解析和提前返回**

将 `main()` 替换为：

```python
def main() -> int:
    parser = argparse.ArgumentParser(description="OpenCortex 启动器")
    parser.add_argument(
        "--rebuild-only",
        action="store_true",
        help="仅重建向量索引，不启动 Streamlit",
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    cfg = read_runtime_config()

    try:
        validate_runtime(cfg)
        rebuild_index(cfg)
    except Exception as exc:
        print(f"[OpenCortex] 启动失败: {exc}", flush=True)
        return 1

    if args.rebuild_only:
        print("[OpenCortex] 索引重建完成，退出（--rebuild-only 模式）。", flush=True)
        return 0

    return launch_streamlit(cfg)
```

**Step 3: 手动验证**

```bash
cd .worktrees/feature/docker-deploy
source ../../venv/bin/activate
python start.py --rebuild-only
```

期望：输出索引重建进度，最后一行为 `索引重建完成，退出（--rebuild-only 模式）。`，**不弹出浏览器，进程正常退出**。

**Step 4: Commit**

```bash
git add start.py
git commit -m "功能：start.py 新增 --rebuild-only 参数，只建索引不启服务"
```

---

### Task 2：`app.py` 热加载——索引 mtime 变化自动清缓存

**Files:**
- Modify: `app.py:207-214`（索引存在检测后、`get_vectorstore` 调用前）

**Step 1: 在 `app.py` 第 207 行（`st.stop()` 之后）和第 209 行（`vectorstore = get_vectorstore(...)` 之前）插入热加载逻辑**

原代码（`app.py:202-214`）：

```python
index_file = Path(cfg["persist_dir"]) / "index.faiss"
if not index_file.exists():
    st.error("未检测到向量索引。")
    st.code("python3 start.py", language="bash")
    st.info("请在终端运行上面的命令：先重建索引，再启动页面。")
    st.stop()

vectorstore = get_vectorstore(
    embed_api_key=cfg["embed_api_key"],
    embed_base_url=cfg["embed_base_url"],
    embed_model=cfg["embed_model"],
    persist_dir=cfg["persist_dir"],
)
```

替换为：

```python
index_file = Path(cfg["persist_dir"]) / "index.faiss"
if not index_file.exists():
    st.error("未检测到向量索引。")
    st.code("python3 start.py", language="bash")
    st.info("请在终端运行上面的命令：先重建索引，再启动页面。")
    st.stop()

# 热加载：索引文件 mtime 变化时清除缓存，下次调用自动重新加载
_current_mtime = index_file.stat().st_mtime
if _current_mtime != st.session_state.get("_index_mtime", 0.0):
    get_vectorstore.clear()
    st.session_state["_index_mtime"] = _current_mtime

vectorstore = get_vectorstore(
    embed_api_key=cfg["embed_api_key"],
    embed_base_url=cfg["embed_base_url"],
    embed_model=cfg["embed_model"],
    persist_dir=cfg["persist_dir"],
)
```

**Step 2: 手动验证（热加载路径）**

1. 启动 Streamlit：`streamlit run app.py`
2. 访问页面，确认正常加载
3. 重新执行 `python start.py --rebuild-only`
4. 刷新浏览器页面
5. 期望：页面刷新后不报错，日志中出现 FAISS 重新加载的日志（无需重启 Streamlit）

**Step 3: Commit**

```bash
git add app.py
git commit -m "功能：app.py 索引热加载，mtime 变化时自动清除 cache_resource"
```

---

### Task 3：新建 `Dockerfile`

**Files:**
- Create: `Dockerfile`

**Step 1: 创建文件，内容如下**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.port=8501"]
```

**Step 2: 构建镜像验证**

```bash
docker build -t opencortex-test .
```

期望：构建成功，最后一行 `Successfully tagged opencortex-test:latest`（或类似输出），无 ERROR。

**Step 3: 清理测试镜像**

```bash
docker rmi opencortex-test
```

**Step 4: Commit**

```bash
git add Dockerfile
git commit -m "部署：新增 Dockerfile，基于 python:3.12-slim"
```

---

### Task 4：新建 `docker-compose.yml`

**Files:**
- Create: `docker-compose.yml`

**Step 1: 创建文件，内容如下**

```yaml
services:
  app:
    build: .
    ports:
      - "${APP_PORT:-8501}:8501"
    env_file: .env
    volumes:
      - ${LOCAL_DOCS_DIR:-./wechat_export}:/data/docs:ro
      - ${CHROMA_PERSIST_DIR:-./data/index}:/data/index
    environment:
      LOCAL_DOCS_DIR: /data/docs
      CHROMA_PERSIST_DIR: /data/index
    restart: unless-stopped
```

> **说明：**
> - `env_file: .env`：将 `.env` 中所有变量加载为容器环境变量（含 API Key）
> - `volumes` 中的 `${VAR}` 由宿主机 `.env` 插值，将宿主机路径挂载到容器固定路径
> - `environment:` 覆盖容器内的路径变量，指向挂载点（`/data/docs`、`/data/index`）
> - Docker Compose 读取项目目录的 `.env` 文件做 compose 文件变量插值，无需额外配置

**Step 2: 验证 compose 配置合法**

```bash
docker compose config
```

期望：输出展开后的配置，无 `error` 或 `warning`。

**Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "部署：新增 docker-compose.yml，支持内网多用户服务"
```

---

### Task 5：新建 `.dockerignore`，更新 `.env.example`

**Files:**
- Create: `.dockerignore`
- Modify: `.env.example`

**Step 1: 创建 `.dockerignore`**

```
venv/
.env
wechat_export/
.worktrees/
__pycache__/
*.pyc
*.pyo
.git/
data/
docs/plans/
```

**Step 2: 在 `.env.example` 末尾补充 Docker 部署说明注释**

在文件末尾现有内容后追加：

```bash
# ── Docker 部署说明 ─────────────────────────────────────────────────────
# docker-compose.yml 使用上面的 LOCAL_DOCS_DIR 和 CHROMA_PERSIST_DIR 作为
# 宿主机挂载路径（volume source），容器内固定使用 /data/docs 和 /data/index。
# 请将两个路径设为宿主机上的**绝对路径**或相对于 docker-compose.yml 的路径。
#
# 首次部署流程：
#   docker compose build
#   docker compose run --rm app python start.py --rebuild-only
#   docker compose up -d
#
# 文档更新后热加载（无需重启容器）：
#   docker compose exec app python start.py --rebuild-only
```

**Step 3: 验证**

```bash
docker compose build --no-cache 2>&1 | tail -5
```

期望：构建成功，`venv/`、`.env`、`wechat_export/` 不被复制进镜像（可用 `docker run --rm opencortex-... ls /app` 确认无这些目录）。

**Step 4: Commit**

```bash
git add .dockerignore .env.example
git commit -m "部署：新增 .dockerignore，补充 .env.example Docker 部署说明"
```

---

### Task 6：端到端验证

**Step 1: 构建镜像**

```bash
docker compose build
```

**Step 2: 首次建索引**

```bash
docker compose run --rm app python start.py --rebuild-only
```

期望：输出索引进度条，最后一行为 `索引重建完成，退出（--rebuild-only 模式）。`，容器退出码 0。

**Step 3: 启动服务**

```bash
docker compose up -d
docker compose logs -f
```

期望：日志出现 `Network URL: http://0.0.0.0:8501`，无 ERROR。

**Step 4: 访问页面**

浏览器访问 `http://localhost:8501`，确认：
- 页面正常加载，显示 OpenCortex 标题和渐变效果
- 输入框可用，发送一条问题可以获得回答

**Step 5: 验证热加载**

```bash
# 不重启容器，直接重建索引
docker compose exec app python start.py --rebuild-only
```

浏览器刷新页面，确认：
- 页面不报错
- 可以正常提问

**Step 6: 最终 Commit**

```bash
git add .
git commit -m "部署：Docker 多用户部署完成，支持热加载"
```
