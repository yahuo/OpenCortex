"""LLM-based reranker for retrieved hits.

Runs after RRF fusion to reorder the top candidates by relevance using the
configured cloud LLM. Falls back to the original ordering on any failure so the
pipeline never breaks because of a rerank issue.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any


def _core():
    import ragbot

    return ragbot


_RERANK_PROMPT_HEADER = (
    "你是检索结果的相关性评审。给定一个用户问题与若干候选片段，对每个候选打 0-10 的"
    "相关性分数（10 = 直接、完整回答问题，0 = 完全无关）。只输出 JSON："
    '{"rankings": [{"index": <候选索引>, "score": <0-10 浮点>, "relevant": <true/false>}]}'
    "。不要给出任何额外文字、解释或代码块。索引必须使用候选片段前的编号。"
)


def _truncate_snippet(snippet: str, max_chars: int) -> str:
    text = snippet.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _build_prompt(question: str, hits: list[Any], max_chars: int) -> str:
    core = _core()
    lines: list[str] = [_RERANK_PROMPT_HEADER, "", f"问题: {question.strip()}", "", "候选片段:"]
    for index, hit in enumerate(hits):
        location = core._line_range_label(hit.line_start, hit.line_end)
        label = f"{hit.source}:{location}" if location else hit.source
        snippet = _truncate_snippet(hit.snippet, max_chars)
        lines.append(f"[{index}] ({label}) {snippet}")
    lines.append("")
    lines.append('请按上述格式返回 JSON。')
    return "\n".join(lines)


def _parse_rerank_payload(text: str, hit_count: int) -> list[dict[str, Any]]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    match = re.search(r"\{.*\}", stripped, re.S)
    if not match:
        raise ValueError("rerank output missing JSON object")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("rerank output is not an object")
    rankings = data.get("rankings")
    if not isinstance(rankings, list):
        raise ValueError("rerank payload missing 'rankings' list")
    parsed: list[dict[str, Any]] = []
    for entry in rankings:
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= hit_count:
            continue
        try:
            score = float(entry.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        relevant_raw = entry.get("relevant", True)
        if isinstance(relevant_raw, str):
            relevant = relevant_raw.strip().lower() not in {"false", "0", "no"}
        else:
            relevant = bool(relevant_raw)
        parsed.append({"index": index, "score": score, "relevant": relevant})
    return parsed


def llm_rerank(
    question: str,
    hits: list[Any],
    *,
    llm_api_key: str,
    llm_model: str,
    llm_base_url: str,
    top_n: int | None = None,
    keep: int | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Rerank `hits` using the configured cloud LLM.

    Returns `(reranked_hits, trace)`. `trace["status"]` is one of
    `"ok" | "disabled" | "no_credentials" | "no_hits" | "fallback"`.
    """
    core = _core()
    trace: dict[str, Any] = {
        "status": "ok",
        "model": llm_model,
        "candidates_in": len(hits),
        "kept": min(len(hits), keep) if keep is not None else len(hits),
        "scores": [],
        "latency_ms": 0,
    }

    if not hits:
        trace["status"] = "no_hits"
        return hits, trace

    if not core._rerank_enabled():
        trace["status"] = "disabled"
        return hits[: keep or len(hits)], trace

    if not llm_api_key.strip() or not llm_base_url.strip():
        trace["status"] = "no_credentials"
        return hits[: keep or len(hits)], trace

    effective_top_n = top_n or core._rerank_top_n()
    candidates = list(hits[:effective_top_n])
    tail = list(hits[effective_top_n:])
    keep_count = keep if keep is not None else len(hits)
    model = core._rerank_model(llm_model)
    trace["model"] = model
    trace["candidates_in"] = len(candidates)

    prompt = _build_prompt(question, candidates, core.RERANK_SNIPPET_MAX_CHARS)
    started = time.perf_counter()
    try:
        llm = core.make_llm(
            api_key=llm_api_key,
            model=model,
            base_url=llm_base_url,
            temperature=0.0,
        )
        response = llm.invoke(prompt)
        content = getattr(response, "content", "") or ""
        rankings = _parse_rerank_payload(str(content), len(candidates))
    except Exception as exc:  # noqa: BLE001 — fallback is the whole point
        trace["status"] = "fallback"
        trace["error"] = f"{type(exc).__name__}: {exc}"
        trace["latency_ms"] = int((time.perf_counter() - started) * 1000)
        trace["kept"] = min(len(hits), keep_count)
        return hits[:keep_count], trace

    trace["latency_ms"] = int((time.perf_counter() - started) * 1000)

    if not rankings:
        trace["status"] = "fallback"
        trace["error"] = "empty rerank rankings"
        trace["kept"] = min(len(hits), keep_count)
        return hits[:keep_count], trace

    relevant_entries: list[tuple[int, float]] = []
    irrelevant_indexes: list[int] = []
    seen_relevant: set[int] = set()
    seen_irrelevant: set[int] = set()
    for entry in sorted(rankings, key=lambda item: item["score"], reverse=True):
        index = entry["index"]
        if entry["relevant"]:
            if index in seen_relevant:
                continue
            seen_relevant.add(index)
            relevant_entries.append((index, entry["score"]))
        else:
            if index in seen_irrelevant or index in seen_relevant:
                continue
            seen_irrelevant.add(index)
            irrelevant_indexes.append(index)

    ordered: list[Any] = [candidates[index] for index, _ in relevant_entries]
    seen_indexes: set[int] = set(seen_relevant)
    for index, candidate in enumerate(candidates):
        if index in seen_indexes:
            continue
        if index in seen_irrelevant:
            continue
        ordered.append(candidate)
        seen_indexes.add(index)
    for index in irrelevant_indexes:
        if index in seen_indexes:
            continue
        ordered.append(candidates[index])
        seen_indexes.add(index)

    ordered.extend(tail)
    scored_indexes = relevant_entries

    trace["scores"] = [
        {"index": idx, "score": score, "source": candidates[idx].source}
        for idx, score in scored_indexes[:keep_count]
    ]
    final = ordered[:keep_count]
    trace["kept"] = len(final)
    return final, trace
