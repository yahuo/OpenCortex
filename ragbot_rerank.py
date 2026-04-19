from __future__ import annotations

import time
from typing import Any


def _core():
    import ragbot

    return ragbot


RERANK_STATUS_OK = "ok"
RERANK_STATUS_DISABLED = "disabled"
RERANK_STATUS_NO_CREDENTIALS = "no_credentials"
RERANK_STATUS_NO_HITS = "no_hits"
RERANK_STATUS_FALLBACK = "fallback"


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
    lines.append("请按上述格式返回 JSON。")
    return "\n".join(lines)


def _parse_rerank_payload(text: str, hit_count: int) -> list[dict[str, Any]]:
    core = _core()
    data = core._extract_json_blob(text)
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
    core = _core()
    keep_count = keep if keep is not None else len(hits)
    trace: dict[str, Any] = {
        "status": RERANK_STATUS_OK,
        "model": llm_model,
        "candidates_in": 0,
        "kept": 0,
        "scores": [],
        "latency_ms": 0,
    }

    if not hits:
        trace["status"] = RERANK_STATUS_NO_HITS
        return hits, trace

    if not core._rerank_enabled():
        trace["status"] = RERANK_STATUS_DISABLED
        out = hits[:keep_count]
        trace["kept"] = len(out)
        return out, trace

    if not llm_api_key.strip() or not llm_base_url.strip():
        trace["status"] = RERANK_STATUS_NO_CREDENTIALS
        out = hits[:keep_count]
        trace["kept"] = len(out)
        return out, trace

    effective_top_n = top_n or core._rerank_top_n()
    candidates = list(hits[:effective_top_n])
    tail = list(hits[effective_top_n:])
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
        content = core._chunk_to_text(response)
        rankings = _parse_rerank_payload(content, len(candidates))
    except Exception as exc:  # noqa: BLE001
        trace["status"] = RERANK_STATUS_FALLBACK
        trace["error"] = f"{type(exc).__name__}: {exc}"
        trace["latency_ms"] = int((time.perf_counter() - started) * 1000)
        out = hits[:keep_count]
        trace["kept"] = len(out)
        return out, trace

    trace["latency_ms"] = int((time.perf_counter() - started) * 1000)

    if not rankings:
        trace["status"] = RERANK_STATUS_FALLBACK
        trace["error"] = "empty rerank rankings"
        out = hits[:keep_count]
        trace["kept"] = len(out)
        return out, trace

    relevant_entries: list[tuple[int, float]] = []
    irrelevant_indexes: list[int] = []
    seen: set[int] = set()
    for entry in sorted(rankings, key=lambda item: item["score"], reverse=True):
        index = entry["index"]
        if index in seen:
            continue
        seen.add(index)
        if entry["relevant"]:
            relevant_entries.append((index, entry["score"]))
        else:
            irrelevant_indexes.append(index)

    ordered: list[Any] = [candidates[index] for index, _ in relevant_entries]
    for index, candidate in enumerate(candidates):
        if index in seen:
            continue
        ordered.append(candidate)
    for index in irrelevant_indexes:
        ordered.append(candidates[index])

    ordered.extend(tail)

    trace["scores"] = [
        {"index": idx, "score": score, "source": candidates[idx].source}
        for idx, score in relevant_entries[:keep_count]
    ]
    final = ordered[:keep_count]
    trace["kept"] = len(final)
    return final, trace
