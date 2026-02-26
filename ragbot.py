#!/usr/bin/env python3
"""
ragbot.py — 微信聊天记录 RAG 问答核心模块

架构：
  OpenAI 兼容 Embedding API → FAISS (向量存储) → 任意 OpenAI 兼容 LLM (生成)

微信 MD 文件格式（由 export_group.py 生成）：
  **发送者** `[YYYY-MM-DD HH:MM:SS]`
  消息内容文本（可能多行）

使用方式：
  1. 首次运行 build_vectorstore() 建立索引
  2. 后续用 load_vectorstore() 加载已有索引
  3. 调用 ask() 提问并获取带引用的答案
"""
from __future__ import annotations

import os
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
DEFAULT_LLM_MODEL = "gemini-2.0-flash"
TIME_WINDOW_MINUTES = 30
DEFAULT_FAISS_DIR = str(Path.home() / "wechat_rag_db")
DEFAULT_TOP_K = 6


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
# 微信 MD 文件解析 & 分块
#
# 实际格式（由 export_group.py 生成）：
#   **发送者昵称** `[2025-02-20 14:29:18]`
#   消息内容（一行或多行）
#   （空行分隔下一条消息）
# ─────────────────────────────────────────────────────────

# 匹配消息头行: **发送者** `[YYYY-MM-DD HH:MM:SS]`
_HEADER_RE = re.compile(
    r"^\*\*(.+?)\*\*\s+`\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]`"
)


def _parse_md_messages(md_path: Path) -> list[dict]:
    """解析微信 MD 文件中的每条消息，返回结构化列表"""
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

    for line in md_path.read_text(encoding="utf-8").splitlines():
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


def _build_documents(md_dir: str) -> list[Document]:
    """将 md_dir 下所有 MD 文件解析并转换为 LangChain Document 列表"""
    documents = []
    md_files = sorted(Path(md_dir).glob("*.md"))

    for md_path in md_files:
        messages = _parse_md_messages(md_path)
        chunks = _chunk_by_time_window(messages)

        for idx, chunk in enumerate(chunks):
            lines = []
            for msg in chunk:
                ts_str = msg["timestamp"].strftime("%H:%M") if msg["timestamp"] else "?"
                lines.append(f"[{ts_str}] {msg['sender']}: {msg['content']}")

            page_content = "\n".join(lines)
            if not page_content.strip():
                continue

            timestamps = [m["timestamp"] for m in chunk if m.get("timestamp")]
            start_t = timestamps[0].strftime("%Y-%m-%d %H:%M") if timestamps else ""
            end_t = timestamps[-1].strftime("%H:%M") if timestamps else ""

            documents.append(Document(
                page_content=page_content,
                metadata={
                    "source": md_path.name,
                    "chunk_index": idx,
                    "start_time": start_t,
                    "end_time": end_t,
                    "time_range": f"{start_t} ~ {end_t}" if start_t else "",
                },
            ))

    return documents


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
    md_dir: Markdown 文件所在目录
    persist_dir: FAISS 索引持久化目录
    """
    def _cb(cur, total, msg):
        if progress_callback:
            progress_callback(cur, total, msg)

    _cb(0, 3, "正在解析 Markdown 文件...")
    documents = _build_documents(md_dir)
    total = len(documents)

    if total == 0:
        raise ValueError(f"在 {md_dir} 中未找到任何可解析的对话片段，请确认 MD 文件格式正确。")

    _cb(1, 3, f"解析完毕，共 {total} 个对话片段，正在向量化（可能需要数分钟）...")

    embeddings = make_embeddings(api_key=embed_api_key, base_url=embed_base_url, model=embed_model)

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
        _cb(1, 3, f"向量化中 {done}/{total}...")

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(persist_path))  # type: ignore[union-attr]
    _cb(3, 3, f"✅ 索引构建完成，已保存到 {persist_dir}")

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
_PROMPT_TEMPLATE = """你是一个私人助手，专门帮用户回顾和分析他们的微信群聊天记录。

以下是与用户问题最相关的历史聊天片段：

{context}

---
请根据以上聊天记录回答用户的问题。要求：
1. 用中文回答，语言简洁自然
2. 如果聊天记录中没有相关信息，如实说明不确定
3. 不要虚构聊天内容

用户问题：{question}"""


def ask(
    question: str,
    vectorstore: FAISS,
    llm_api_key: str,
    llm_model: str,
    llm_base_url: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    检索 + 生成，返回：
      {
        "answer": str,
        "sources": [{"source": str, "time_range": str, "snippet": str}, ...]
      }
    """
    # 1. 检索
    results = vectorstore.similarity_search(question, k=top_k)

    # 2. 构造上下文
    context_parts = []
    for i, doc in enumerate(results):
        meta = doc.metadata
        header = f"[片段{i+1}] 来自《{meta.get('source', '')}》{meta.get('time_range', '')}"
        context_parts.append(f"{header}\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    # 3. 调用 LLM
    llm = make_llm(api_key=llm_api_key, model=llm_model, base_url=llm_base_url)
    prompt = _PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    # 4. 整理引用列表
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

    return {"answer": answer, "sources": sources}
