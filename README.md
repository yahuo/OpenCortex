# OpenCortex

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/vector--store-FAISS-green.svg)](https://github.com/facebookresearch/faiss)

**A lightweight, local-first RAG chatbot.** Point it at a local directory, let it index your documents, then chat with them using any OpenAI-compatible LLM — all running on your own machine.

[中文文档](README.zh-CN.md)

---

## Features

- **Zero-config single-page UI** — Streamlit, no settings page needed
- **Recursive indexing** — supports `.md`, `.txt`, `.json`, `.yaml`, `.csv`, `.rst`, `.log`, `.docx`, `.xlsx`, `.pdf`, and more
- **WeChat export support** — natively parses WeChat-exported Markdown with time-window chunking
- **Pluggable LLM & embeddings** — any OpenAI-compatible API: SiliconFlow, Gemini, DeepSeek, Kimi, GLM, etc.
- **Local FAISS vector store** — no cloud dependency, data stays on your machine
- **Hot-reload** — update your docs and trigger re-indexing without restarting the server
- **Docker deployment** — serve the same knowledge base to multiple users on a LAN

---

## Architecture

```
Local docs directory
      │
      ▼ ragbot.py — parse, chunk, embed → FAISS index (persisted locally)
      │
      ▼ app.py — Streamlit UI, loads FAISS, handles chat
      │
      ▼ Any OpenAI-compatible LLM (streaming)
```

`start.py` orchestrates the flow: rebuild index → launch web server.

---

## Quick Start

### Local (recommended for personal use)

Requires Python 3.12.

```bash
git clone https://github.com/yahuo/OpenCortex.git
cd WechatLLM

python3.12 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — set LOCAL_DOCS_DIR, EMBED_API_KEY, LLM_API_KEY at minimum
```

Start the app (rebuilds index, then opens the web server):

```bash
source venv/bin/activate
python3 start.py
```

Then open `http://localhost:8501` in your browser.

### Docker (LAN multi-user)

Requires Docker and Docker Compose v2.

```bash
cp .env.example .env
# Edit .env — set LOCAL_DOCS_DIR, EMBED_API_KEY, LLM_API_KEY
# Use absolute paths for LOCAL_DOCS_DIR and CHROMA_PERSIST_DIR
```

**First deployment:**

```bash
docker compose build
docker compose run --rm app python start.py --rebuild-only   # build index
docker compose up -d                                          # start service
```

Access via `http://<server-ip>:8501`.

**Reload docs without restarting:**

```bash
docker compose exec app python start.py --rebuild-only
# Then refresh the browser — new index loads automatically
```

**Common ops:**

```bash
docker compose logs -f    # tail logs
docker compose down       # stop
docker compose up -d      # restart
```

---

## Configuration

Copy `.env.example` to `.env` and fill in the required values.

| Variable | Required | Default | Description |
|---|:---:|---|---|
| `LOCAL_DOCS_DIR` | ✅ | `./docs` | Directory to index (recursively) |
| `EMBED_API_KEY` | ✅ | — | API key for embedding service |
| `EMBED_BASE_URL` | | `https://api.siliconflow.cn/v1` | Embedding API base URL |
| `EMBED_MODEL` | | `BAAI/bge-m3` | Embedding model name |
| `LLM_API_KEY` | ✅ | — | API key for the LLM |
| `LLM_BASE_URL` | | `https://generativelanguage.googleapis.com/v1beta/openai/` | LLM API base URL |
| `LLM_MODEL` | | `gemini-2.0-flash` | LLM model name |
| `CHROMA_PERSIST_DIR` | | `~/wechat_rag_db` | Directory to persist the FAISS index |
| `APP_HOST` | | `127.0.0.1` | Server bind address |
| `APP_PORT` | | `8501` | Server port |

**Compatible LLM providers** (set `LLM_BASE_URL` accordingly):

| Provider | Base URL |
|---|---|
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| DeepSeek | `https://api.deepseek.com/v1` |
| Kimi (Moonshot) | `https://api.moonshot.cn/v1` |
| GLM (Zhipu) | `https://open.bigmodel.cn/api/paas/v4/` |
| SiliconFlow | `https://api.siliconflow.cn/v1` |

---

## Supported File Types

| Extension | Chunking strategy |
|---|---|
| `.md`, `.markdown`, `.mdx` | WeChat format → time-window chunks; otherwise fixed-size |
| `.txt`, `.rst`, `.log` | Fixed-size with overlap |
| `.csv`, `.json`, `.yaml`, `.yml` | Fixed-size with overlap |
| `.docx`, `.xlsx`, `.pdf` | markitdown → Markdown → fixed-size with overlap |

---

## File Structure

```
OpenCortex/
├── start.py          # Launcher: rebuild index → start web server
├── app.py            # Streamlit single-page UI
├── ragbot.py         # Core: chunking, embedding, FAISS, RAG Q&A
├── Dockerfile        # Container image
├── docker-compose.yml# Multi-user deployment
├── requirements.txt  # Python dependencies
└── .env.example      # Environment variable template
```

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create a branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Open a Pull Request

---

## License

[MIT](LICENSE) © 2024 OpenCortex contributors
