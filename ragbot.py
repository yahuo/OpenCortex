#!/usr/bin/env python3
"""
ragbot.py — 通用本地文档 RAG 问答核心模块

架构：
  OpenAI 兼容 Embedding API → FAISS (向量存储) → 任意 OpenAI 兼容 LLM (生成)

使用方式：
  1. 首次运行 build_vectorstore() 建立索引
  2. 后续用 load_vectorstore() 加载已有索引
  3. 调用 ask_stream() 提问并获取流式答案
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ─────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
TIME_WINDOW_MINUTES = 30
DEFAULT_FAISS_DIR = str(Path.home() / "wechat_rag_db")
DEFAULT_TOP_K = 6
SUPPORTED_TEXT_SUFFIXES = {
    ".md",
    ".markdown",
    ".mdx",
    ".txt",
    ".rst",
    ".log",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
}
GENERIC_CHUNK_SIZE = 1200
GENERIC_CHUNK_OVERLAP = 180


# ─────────────────────────────────────────────────────────
# Embedding 工厂
# ─────────────────────────────────────────────────────────
def make_embeddings(
    api_key: str,
    base_url: str = SILICONFLOW_BASE_URL,
    model: str = DEFAULT_EMBED_MODEL,
) -> OpenAIEmbeddings:
    """创建向量化器（兼容任意 OpenAI 格式 Embedding 接口）"""
    return OpenAIEmbeddings(
        model=model,
        base_url=base_url,
        api_key=api_key,
        check_embedding_ctx_length=False,
    )


# ─────────────────────────────────────────────────────────
# LLM 工厂（通用 OpenAI 兼容）
# ─────────────────────────────────────────────────────────
def make_llm(
    api_key: str,
    model: str,
    base_url: str,
    temperature: float = 0.3,
) -> ChatOpenAI:
    """
    创建 LLM 客户端，支持任何 OpenAI 兼容接口：
      - Gemini: base_url=https://generativelanguage.googleapis.com/v1beta/openai/
      - Kimi:   base_url=https://api.moonshot.cn/v1
      - GLM:    base_url=https://open.bigmodel.cn/api/paas/v4/
      - DeepSeek: base_url=https://api.deepseek.com/v1
    """
    return ChatOpenAI(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
    )


# ─────────────────────────────────────────────────────────
# 微信样式 MD 文件解析 & 分块
#
# 兼容格式示例：
#   **发送者昵称** `[2025-02-20 14:29:18]`
#   消息内容（一行或多行）
#   （空行分隔下一条消息）
# ─────────────────────────────────────────────────────────

# 匹配消息头行: **发送者** `[YYYY-MM-DD HH:MM:SS]`
_HEADER_RE = re.compile(
    r"^\*\*(.+?)\*\*\s+`\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]`"
)


def _parse_md_messages_text(content: str) -> list[dict]:
    """解析微信 MD 文本中的每条消息，返回结构化列表"""
    messages = []
    current_sender: str | None = None
    current_ts: datetime | None = None
    current_lines: list[str] = []

    def flush():
        if current_sender is not None:
            content = "\n".join(current_lines).strip()
            if content:
                messages.append({
                    "sender": current_sender,
                    "timestamp": current_ts,
                    "content": content,
                })

    for line in content.splitlines():
        m = _HEADER_RE.match(line)
        if m:
            flush()
            current_sender = m.group(1).strip()
            try:
                current_ts = datetime.strptime(m.group(2).strip(), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                current_ts = None
            current_lines = []
        elif current_sender is not None:
            # 跳过 Markdown 分隔符和元数据行
            stripped = line.strip()
            if stripped and stripped not in ("---",) and not stripped.startswith("> 导出时间"):
                current_lines.append(stripped)

    flush()
    return messages


def _chunk_by_time_window(
    messages: list[dict],
    window_minutes: int = TIME_WINDOW_MINUTES,
) -> list[list[dict]]:
    """按时间窗口将消息列表分成多个 chunk（避免跨主题混淆）"""
    if not messages:
        return []

    chunks, current_chunk = [], []
    window_start: datetime | None = None

    for msg in messages:
        ts = msg.get("timestamp")
        if ts is None:
            current_chunk.append(msg)
            continue

        if window_start is None:
            window_start = ts

        if ts - window_start <= timedelta(minutes=window_minutes):
            current_chunk.append(msg)
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [msg]
            window_start = ts

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _iter_supported_files(source_dir: Path) -> list[Path]:
    """递归列出 source_dir 下支持的文本文件"""
    files: list[Path] = []
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_TEXT_SUFFIXES:
            files.append(path)
    return sorted(files)


def _split_generic_text(
    text: str,
    chunk_size: int = GENERIC_CHUNK_SIZE,
    overlap: int = GENERIC_CHUNK_OVERLAP,
) -> list[str]:
    """按固定窗口切分通用文本，保留一定重叠以维持上下文"""
    clean_text = text.replace("\x00", " ").strip()
    if not clean_text:
        return []

    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for start in range(0, len(clean_text), step):
        part = clean_text[start : start + chunk_size].strip()
        if part:
            chunks.append(part)
        if start + chunk_size >= len(clean_text):
            break
    return chunks


def _build_documents(source_dir: str) -> tuple[list[Document], list[Path]]:
    """将 source_dir 下通用文本文件转换为 LangChain Document 列表"""
    root = Path(source_dir)
    files = _iter_supported_files(root)
    documents: list[Document] = []

    for file_path in files:
        rel_path = str(file_path.relative_to(root))
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            # 单文件异常不应中断整体索引流程
            continue
        if not text.strip():
            continue

        # 若匹配微信导出 MD 格式，则使用时间窗口分块，保留时间元数据
        if file_path.suffix.lower() in {".md", ".markdown"}:
            messages = _parse_md_messages_text(text)
            if messages:
                chunks = _chunk_by_time_window(messages)
                for idx, chunk in enumerate(chunks):
                    lines = []
                    for msg in chunk:
                        ts_str = (
                            msg["timestamp"].strftime("%H:%M")
                            if msg["timestamp"]
                            else "?"
                        )
                        lines.append(f"[{ts_str}] {msg['sender']}: {msg['content']}")

                    page_content = "\n".join(lines).strip()
                    if not page_content:
                        continue

                    timestamps = [m["timestamp"] for m in chunk if m.get("timestamp")]
                    start_t = (
                        timestamps[0].strftime("%Y-%m-%d %H:%M")
                        if timestamps
                        else ""
                    )
                    end_t = (
                        timestamps[-1].strftime("%H:%M")
                        if timestamps
                        else ""
                    )

                    documents.append(
                        Document(
                            page_content=page_content,
                            metadata={
                                "source": rel_path,
                                "chunk_index": idx,
                                "start_time": start_t,
                                "end_time": end_t,
                                "time_range": f"{start_t} ~ {end_t}" if start_t else "",
                            },
                        )
                    )
                continue

        for idx, chunk_text in enumerate(_split_generic_text(text)):
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source": rel_path,
                        "chunk_index": idx,
                        "time_range": f"chunk {idx + 1}",
                    },
                )
            )

    return documents, files


# ─────────────────────────────────────────────────────────
# 向量库操作（使用 FAISS，兼容 Python 3.14）
# ─────────────────────────────────────────────────────────
def build_vectorstore(
    md_dir: str,
    embed_api_key: str,
    embed_base_url: str = SILICONFLOW_BASE_URL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    persist_dir: str = DEFAULT_FAISS_DIR,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> FAISS:
    """
    构建（或重建）FAISS 向量索引并持久化到本地。
    md_dir: 文本源目录（会递归读取支持的文本文件）
    persist_dir: FAISS 索引持久化目录
    """
    def _cb(cur, total, msg):
        if progress_callback:
            progress_callback(cur, total, msg)

    _cb(0, 1, "正在扫描目录并解析文本文件...")
    documents, source_files = _build_documents(md_dir)
    total = len(documents)

    if total == 0:
        raise ValueError(
            f"在 {md_dir} 中未找到可索引文本。支持后缀: {', '.join(sorted(SUPPORTED_TEXT_SUFFIXES))}"
        )

    total_steps = total + 2
    _cb(0, total_steps, f"解析完毕，共 {len(source_files)} 个文件，{total} 个分片。")

    embeddings = make_embeddings(
        api_key=embed_api_key, base_url=embed_base_url, model=embed_model
    )

    # 分批向量化，避免超出 API 限制
    BATCH_SIZE = 50
    vectorstore: FAISS | None = None

    for i in range(0, total, BATCH_SIZE):
        batch = documents[i: i + BATCH_SIZE]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        done = min(i + BATCH_SIZE, total)
        _cb(done, total_steps, f"向量化中 {done}/{total}...")

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(persist_path))  # type: ignore[union-attr]
    _cb(total + 1, total_steps, "正在写入索引清单...")

    # 写入索引文件清单，供 UI 展示"当前索引了哪些文件"
    manifest = {
        "build_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "embed_model": embed_model,
        "total_chunks": total,
        "files": [],
    }
    root = Path(md_dir)
    for file_path in source_files:
        stat = file_path.stat()
        rel_path = str(file_path.relative_to(root))
        chunks = sum(
            1 for d in documents if d.metadata.get("source") == rel_path
        )
        if chunks == 0:
            continue
        manifest["files"].append({
            "name": rel_path,
            "size_kb": round(stat.st_size / 1024, 1),
            "mtime": datetime.fromtimestamp(stat.st_mtime).strftime(
                "%Y-%m-%d %H:%M"
            ),
            "chunks": chunks,
        })
    (persist_path / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _cb(total + 2, total_steps, f"✅ 索引构建完成，已保存到 {persist_dir}")

    return vectorstore  # type: ignore[return-value]


def load_vectorstore(
    embed_api_key: str,
    embed_base_url: str = SILICONFLOW_BASE_URL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    persist_dir: str = DEFAULT_FAISS_DIR,
) -> FAISS | None:
    """加载已存在的 FAISS 向量索引，目录不存在则返回 None"""
    index_file = Path(persist_dir) / "index.faiss"
    if not index_file.exists():
        return None
    embeddings = make_embeddings(api_key=embed_api_key, base_url=embed_base_url, model=embed_model)
    return FAISS.load_local(
        str(persist_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ─────────────────────────────────────────────────────────
# RAG 问答
# ─────────────────────────────────────────────────────────
_PROMPT_TEMPLATE = """你是一个私人知识助手。以下参考资料供你回答时使用：

{context}

---
要求：
1. 直接给出答案，像一个领域专家一样回答
2. 不要提及“片段X”“来源文件”等内部检索过程
3. 如果信息不足以回答，直接说不确定，不要虚构
4. 用中文回答，语言简洁专业

用户问题：{question}"""


def _build_context_and_sources(results: list[Document]) -> tuple[str, list[dict]]:
    """由检索结果构建上下文与引用列表"""
    # 1. 检索
    context_parts = []
    for i, doc in enumerate(results):
        meta = doc.metadata
        source = meta.get("source", "")
        time_range = meta.get("time_range", "")
        header = f"[片段{i+1}] 来自《{source}》"
        if time_range:
            header += f" {time_range}"
        context_parts.append(f"{header}\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    # 2. 整理引用列表
    sources = []
    seen: set[tuple] = set()
    for doc in results:
        meta = doc.metadata
        key = (meta.get("source", ""), meta.get("time_range", ""))
        if key not in seen:
            seen.add(key)
            snippet = doc.page_content
            sources.append({
                "source": meta.get("source", ""),
                "time_range": meta.get("time_range", ""),
                "snippet": snippet[:150] + "..." if len(snippet) > 150 else snippet,
            })

    return context, sources


def _chunk_to_text(chunk) -> str:
    """将 LangChain 流式 chunk 统一转换为字符串"""
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def ask_stream(
    question: str,
    vectorstore: FAISS,
    llm_api_key: str,
    llm_model: str,
    llm_base_url: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    检索 + 流式生成，返回：
      {
        "answer_stream": Iterator[str],
        "sources": [{"source": str, "time_range": str, "snippet": str}, ...]
      }
    """
    # 1. 检索
    results = vectorstore.similarity_search(question, k=top_k)

    # 2. 构造上下文 + 引用列表
    context, sources = _build_context_and_sources(results)

    # 4. LLM 流式输出
    llm = make_llm(api_key=llm_api_key, model=llm_model, base_url=llm_base_url)
    prompt = _PROMPT_TEMPLATE.format(context=context, question=question)

    def answer_stream():
        for chunk in llm.stream(prompt):
            text = _chunk_to_text(chunk)
            if text:
                yield text

    return {"answer_stream": answer_stream(), "sources": sources}
