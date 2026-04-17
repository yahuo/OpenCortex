#!/usr/bin/env python3
"""诊断脚本：只读扫描 OpenCortex 索引产物，并可选地跑一组探针查询。

用法示例：
  python diagnose.py --skip-probes
  python diagnose.py --num-auto-probes 5 --modes vector,hybrid
  python diagnose.py --probes known_misses.json --json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import ragbot
from ragbot_artifacts import load_index_manifest
from ragbot_runtime import search_bundle_artifact_signature

load_dotenv()


CACHE_KEY_RE = re.compile(r"^section:(.+):(\d+)$")
FILENAME_STOPWORDS = {"index", "main", "init", "util", "test", "tests", "setup"}
HEADING_RE = re.compile(r"^#{1,2}\s+(.+)\s*$")


# ─────────────────────────────────────────────────────────
# 静态审计
# ─────────────────────────────────────────────────────────
def audit_artifacts(persist_dir: Path) -> list[dict[str, Any]]:
    signature = search_bundle_artifact_signature(persist_dir)
    artifacts: list[dict[str, Any]] = []
    for relpath, exists, mtime_ns, size in signature:
        artifacts.append(
            {
                "path": relpath,
                "exists": bool(exists),
                "size_bytes": int(size),
                "mtime": int(mtime_ns),
            }
        )
    return artifacts


def audit_tokens(persist_dir: Path) -> dict[str, Any]:
    cache_path = persist_dir / ragbot.SEMANTIC_EXTRACT_CACHE_FILENAME
    empty = {
        "api_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_sections": 0,
        "failed_sections": 0,
        "avg_total_tokens_per_call": 0.0,
        "current_prompt_version": ragbot.SEMANTIC_PROMPT_VERSION,
        "stale_entries": 0,
        "status_counts": {"ok": 0, "error": 0},
        "top_token_files": [],
        "cache_exists": False,
    }
    if not cache_path.exists():
        return empty
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        empty["cache_exists"] = True
        empty["note"] = "cache file exists but failed to parse"
        return empty

    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, dict):
        entries = {}

    current_prompt_version = ragbot.SEMANTIC_PROMPT_VERSION
    status_counts: Counter[str] = Counter()
    stale = 0
    prompt_total = 0
    completion_total = 0
    total_total = 0
    per_source_tokens: defaultdict[str, int] = defaultdict(int)
    per_source_sections: defaultdict[str, int] = defaultdict(int)

    for key, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or "")
        status_counts[status] += 1
        if str(entry.get("prompt_version") or "") != current_prompt_version:
            stale += 1
        usage = entry.get("usage") if isinstance(entry.get("usage"), dict) else {}
        prompt_total += int(usage.get("prompt_tokens") or 0)
        completion_total += int(usage.get("completion_tokens") or 0)
        total_total += int(usage.get("total_tokens") or 0)

        match = CACHE_KEY_RE.match(str(key))
        source = match.group(1) if match else ""
        if source:
            per_source_tokens[source] += int(usage.get("total_tokens") or 0)
            per_source_sections[source] += 1

    api_calls = int(status_counts.get("ok", 0))
    avg = float(total_total) / api_calls if api_calls else 0.0

    top_files = sorted(
        (
            {
                "source": source,
                "sections": per_source_sections[source],
                "total_tokens": per_source_tokens[source],
            }
            for source in per_source_tokens
        ),
        key=lambda item: item["total_tokens"],
        reverse=True,
    )[:10]

    return {
        "api_calls": api_calls,
        "prompt_tokens": prompt_total,
        "completion_tokens": completion_total,
        "total_tokens": total_total,
        "cached_sections": int(status_counts.get("ok", 0)),
        "failed_sections": int(status_counts.get("error", 0)),
        "avg_total_tokens_per_call": round(avg, 1),
        "current_prompt_version": current_prompt_version,
        "stale_entries": stale,
        "status_counts": dict(status_counts),
        "top_token_files": top_files,
        "cache_exists": True,
    }


def audit_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, list):
        files = []

    chunks_list: list[int] = []
    truncated_files: list[dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        try:
            chunk_count = int(entry.get("chunks") or 0)
        except (TypeError, ValueError):
            chunk_count = 0
        chunks_list.append(chunk_count)
        if entry.get("truncated"):
            truncated_files.append(
                {
                    "name": entry.get("name", ""),
                    "chunks": chunk_count,
                    "original_chunks": int(entry.get("original_chunks") or 0),
                    "size_kb": float(entry.get("size_kb") or 0.0),
                }
            )

    chunks_stats = _percentiles(chunks_list)

    build_config = manifest.get("build_config") if isinstance(manifest, dict) else None
    if not isinstance(build_config, dict):
        build_config = {}
    build_flags = {
        "skip_graph": bool(build_config.get("skip_graph")),
        "skip_semantic": bool(build_config.get("skip_semantic")),
        "skip_wiki": bool(build_config.get("skip_wiki")),
    }

    semantic_stats = manifest.get("semantic_graph_stats") if isinstance(manifest, dict) else {}
    if not isinstance(semantic_stats, dict):
        semantic_stats = {}

    return {
        "file_count": len(files),
        "total_chunks": sum(chunks_list),
        "chunks_per_file": chunks_stats,
        "truncated_files": truncated_files,
        "build_flags": build_flags,
        "search_mode_default": os.getenv("SEARCH_MODE", "hybrid").strip().lower() or "hybrid",
        "semantic_graph_stats": semantic_stats,
    }


def audit_chunking_redflags(
    manifest: dict[str, Any],
    chunk_kind_by_source: dict[str, list[str]] | None,
) -> dict[str, Any]:
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, list):
        files = []

    many_tiny: list[dict[str, Any]] = []
    single_large: list[dict[str, Any]] = []
    generic_fallback: list[dict[str, Any]] = []

    structured_extensions = {".py", ".md", ".markdown", ".json", ".yaml", ".yml", ".toml"}

    for entry in files:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "")
        chunks = int(entry.get("chunks") or 0)
        size_kb = float(entry.get("size_kb") or 0.0)
        suffix = str(entry.get("suffix") or "").lower()

        if chunks > 50 and size_kb > 0 and size_kb / max(chunks, 1) < 2.0:
            many_tiny.append({"name": name, "chunks": chunks, "size_kb": size_kb})

        if chunks == 1 and size_kb > 50:
            single_large.append({"name": name, "chunks": chunks, "size_kb": size_kb})

        if chunk_kind_by_source is not None and suffix in structured_extensions:
            kinds = chunk_kind_by_source.get(name) or []
            if kinds and all(k == "generic" or not k for k in kinds):
                generic_fallback.append({"name": name, "suffix": suffix, "chunks": chunks})

    return {
        "many_tiny_chunks": many_tiny[:15],
        "single_chunk_large": single_large[:15],
        "generic_fallback_with_known_extension": generic_fallback[:15],
    }


def _percentiles(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "max": 0}
    sorted_values = sorted(values)
    return {
        "min": sorted_values[0],
        "p50": int(statistics.median(sorted_values)),
        "p90": sorted_values[max(0, int(len(sorted_values) * 0.9) - 1)],
        "max": sorted_values[-1],
    }


# ─────────────────────────────────────────────────────────
# 探针生成与执行
# ─────────────────────────────────────────────────────────
def build_auto_probes(bundle: Any, num_probes: int, seed: int) -> list[dict[str, Any]]:
    if num_probes <= 0:
        return []
    rng = random.Random(seed)

    target_filename = max(1, int(num_probes * 0.4))
    target_symbol = max(1, int(num_probes * 0.4))
    target_heading = max(0, num_probes - target_filename - target_symbol)

    probes: list[dict[str, Any]] = []

    filename_candidates = sorted(
        (entry for entry in bundle.files if isinstance(entry, dict) and entry.get("name")),
        key=lambda e: float(e.get("size_kb") or 0.0),
        reverse=True,
    )
    seen_stems: set[str] = set()
    for entry in filename_candidates:
        if len(probes) >= target_filename:
            break
        source = entry["name"]
        stem = Path(source).stem
        if not stem or len(stem) < 4 or stem.lower() in FILENAME_STOPWORDS:
            continue
        if stem.lower() in seen_stems:
            continue
        seen_stems.add(stem.lower())
        probes.append(
            {
                "question": stem,
                "kind": "filename",
                "expected_source": source,
            }
        )

    symbol_candidates: list[dict[str, Any]] = []
    seen_symbols: set[str] = set()
    for record in bundle.symbol_index or []:
        if not isinstance(record, dict):
            continue
        kind = str(record.get("kind") or "")
        name = str(record.get("name") or "")
        source = str(record.get("source") or "")
        if kind not in {"class", "function"}:
            continue
        if len(name) < 5 or name in seen_symbols:
            continue
        if not source:
            continue
        seen_symbols.add(name)
        symbol_candidates.append({"name": name, "source": source})

    rng.shuffle(symbol_candidates)
    for record in symbol_candidates[:target_symbol]:
        probes.append(
            {
                "question": record["name"],
                "kind": "symbol",
                "expected_source": record["source"],
            }
        )

    markdown_entries = [
        entry
        for entry in bundle.files
        if isinstance(entry, dict) and str(entry.get("suffix") or "").lower() in {".md", ".markdown"}
    ]
    rng.shuffle(markdown_entries)
    heading_added = 0
    for entry in markdown_entries:
        if heading_added >= target_heading:
            break
        source = entry["name"]
        cache_path = bundle.cache_path_for(source)
        if not cache_path or not cache_path.exists():
            continue
        heading = _first_heading(cache_path)
        if not heading:
            continue
        probes.append(
            {
                "question": heading,
                "kind": "heading",
                "expected_source": source,
            }
        )
        heading_added += 1

    return probes


def _first_heading(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                match = HEADING_RE.match(line.rstrip())
                if match:
                    return match.group(1).strip()
                if fh.tell() > 8192:
                    break
    except OSError:
        return ""
    return ""


def load_user_probes(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"--probes 文件读不出来：{exc}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--probes 文件不是合法 JSON：{exc}")
    if not isinstance(data, list):
        raise SystemExit("--probes 文件顶层必须是 list[{question, expected_source?}]")

    probes: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        probes.append(
            {
                "question": question,
                "kind": "user",
                "expected_source": str(item.get("expected_source") or "") or None,
            }
        )
    return probes


def run_probes(
    bundle: Any,
    probes: list[dict[str, Any]],
    modes: list[str],
    top_k: int,
    allow_llm: bool,
    llm_env: dict[str, str],
) -> dict[str, Any]:
    from ragbot_retrieval import retrieve

    query_results: list[dict[str, Any]] = []
    all_flags: Counter[str] = Counter()
    recall_counts: dict[str, list[int]] = {mode: [0, 0] for mode in modes}

    for probe in probes:
        per_mode: dict[str, dict[str, Any]] = {}
        hybrid_sources: set[str] = set()

        for mode in modes:
            use_llm = allow_llm and mode == "agentic" and bool(llm_env.get("llm_api_key"))
            retrieved = retrieve(
                question=probe["question"],
                search_bundle=bundle,
                kb=None,
                mode=mode,
                top_k=top_k,
                llm_api_key=llm_env["llm_api_key"] if use_llm else "",
                llm_model=llm_env["llm_model"] if use_llm else "",
                llm_base_url=llm_env["llm_base_url"] if use_llm else "",
            )
            hits = list(retrieved.get("hits") or [])
            trace = list(retrieved.get("search_trace") or [])
            step1_trace = trace[0] if trace else {}
            retrievers_count = step1_trace.get("retrievers") if isinstance(step1_trace, dict) else {}
            if not isinstance(retrievers_count, dict):
                retrievers_count = {}

            top_hits: list[dict[str, Any]] = []
            expected_rank: int | None = None
            found_via: str | None = None
            snippet_flags_all: list[str] = []
            sources_in_top: list[str] = []

            normalized_expected = (probe.get("expected_source") or "").strip()
            query_tokens = {tok.lower() for tok in re.split(r"\W+", probe["question"]) if tok}

            for rank, hit in enumerate(hits, start=1):
                source = hit.source
                snippet = hit.snippet or ""
                flags: list[str] = []
                if not snippet.strip():
                    flags.append("empty")
                elif len(snippet) < 60:
                    flags.append("short")
                if query_tokens and not any(tok in snippet.lower() for tok in query_tokens):
                    flags.append("no_query_token")

                for flag in flags:
                    all_flags[flag] += 1
                if flags:
                    snippet_flags_all.extend(f"{hit.match_kind}:{flag}" for flag in flags)

                sources_in_top.append(source)
                top_hits.append(
                    {
                        "source": source,
                        "match_kind": hit.match_kind,
                        "score": round(float(hit.score), 4),
                        "snippet_len": len(snippet),
                        "line_start": hit.line_start,
                        "line_end": hit.line_end,
                        "flags": flags,
                    }
                )
                if normalized_expected and source == normalized_expected and expected_rank is None:
                    expected_rank = rank
                    found_via = hit.match_kind

            mode_entry: dict[str, Any] = {
                "top": top_hits,
                "top_sources": sources_in_top,
                "expected_rank": expected_rank,
                "found_via": found_via,
                "snippet_flags": snippet_flags_all,
                "retrievers_count": {str(k): int(v) for k, v in retrievers_count.items()},
            }
            if mode == "agentic":
                mode_entry["planner_called"] = bool(
                    len(trace) > 1 and trace[1].get("planner_used")
                )
                mode_entry["llm_used"] = use_llm

            per_mode[mode] = mode_entry
            if mode == "hybrid":
                hybrid_sources = set(sources_in_top)

            if normalized_expected:
                recall_counts[mode][1] += 1
                if expected_rank is not None:
                    recall_counts[mode][0] += 1

        overlap = {}
        for mode, data in per_mode.items():
            if mode == "hybrid" or not hybrid_sources:
                continue
            other = set(data["top_sources"])
            union = hybrid_sources | other
            jaccard = (len(hybrid_sources & other) / len(union)) if union else 0.0
            overlap[mode] = {
                "jaccard": round(jaccard, 3),
                "common": len(hybrid_sources & other),
            }

        query_results.append(
            {
                "question": probe["question"],
                "kind": probe["kind"],
                "expected_source": probe.get("expected_source"),
                "modes": per_mode,
                "overlap_with_hybrid": overlap,
            }
        )

    summary = {
        "expected_recall": {
            mode: f"{hits}/{total}" if total else "n/a"
            for mode, (hits, total) in recall_counts.items()
        },
        "snippet_quality": {
            "empty": int(all_flags.get("empty", 0)),
            "short": int(all_flags.get("short", 0)),
            "no_query_token": int(all_flags.get("no_query_token", 0)),
        },
    }

    return {"queries": query_results, "summary": summary, "recall_raw": recall_counts}


# ─────────────────────────────────────────────────────────
# Verdict 规则
# ─────────────────────────────────────────────────────────
def build_verdict(audit: dict[str, Any], probes: dict[str, Any] | None) -> list[str]:
    lines: list[str] = []
    tokens = audit.get("tokens", {})
    manifest_info = audit.get("manifest", {})

    # 规则 0：关键 artifact 缺失
    critical = {
        "symbol_index.jsonl": "AST 符号检索（ast_search）将永远返回空",
        "fulltext_index.json": "BM25 检索（bm25_search）将永远返回空，load_search_bundle 也会直接失败",
    }
    supporting = {
        "document_graph.json": "graph_neighbors 候选扩展失效",
        "entity_graph.json": "entity 候选扩展失效，只能回退到 document_graph/启发式",
    }
    missing_critical: list[str] = []
    missing_supporting: list[str] = []
    for item in audit.get("artifacts", []):
        path = item.get("path", "")
        if path in critical and not item.get("exists"):
            missing_critical.append(f"{path} — {critical[path]}")
        if path in supporting and not item.get("exists"):
            missing_supporting.append(f"{path} — {supporting[path]}")
    if missing_critical:
        lines.append(
            "[关键产物缺失] 以下 artifact 不存在，即便 SEARCH_MODE=hybrid 也只有部分路数能工作："
            + "；".join(missing_critical)
            + "。跑 `python start.py --rebuild-only` 重建会产出这些文件。"
        )
    if missing_supporting:
        lines.append(
            "[辅助产物缺失] " + "；".join(missing_supporting)
            + "。不致命但会拖低 hybrid/agentic 的召回上限。"
        )

    # 规则 1：token 集中
    top_files = tokens.get("top_token_files", []) or []
    total_tokens = int(tokens.get("total_tokens") or 0)
    if top_files and total_tokens > 0:
        top5_sum = sum(int(item.get("total_tokens") or 0) for item in top_files[:5])
        ratio = top5_sum / total_tokens if total_tokens else 0.0
        if ratio > 0.5:
            names = ", ".join(item["source"] for item in top_files[:5])
            lines.append(
                f"[token 集中] top 5 文件占用 {ratio:.0%} 的 build token（{top5_sum:,}/{total_tokens:,}）：{names}。"
                " ragbot_semantic.py:571 的 section 级抽取对每个 SEMANTIC_SECTION_MAX_CHARS=2500 切片跑一次 LLM，"
                "考虑对这些文件按 chunk_kind 跳过或粗粒度化。"
            )

    # 规则 2：缓存陈旧率
    stale = int(tokens.get("stale_entries") or 0)
    cache_ok = int(tokens.get("cached_sections") or 0) + int(tokens.get("failed_sections") or 0)
    if cache_ok > 0 and stale / cache_ok > 0.2:
        lines.append(
            f"[缓存陈旧] {stale}/{cache_ok} cache 条目的 prompt_version 与当前"
            f" '{tokens.get('current_prompt_version')}' 不一致，下次 build 将重跑。"
            " 若未主动改 prompt，检查 ragbot.py:255 的 SEMANTIC_PROMPT_VERSION。"
        )

    # 规则 3：召回失败按探针类型分桶
    if probes:
        misses_by_kind: defaultdict[str, list[str]] = defaultdict(list)
        hybrid_only_recovery = 0
        vector_misses = 0
        user_probes_with_expected = 0
        for query in probes["queries"]:
            expected = query.get("expected_source")
            if not expected:
                continue
            user_probes_with_expected += 1
            mode_entries = query["modes"]
            vector_hit = mode_entries.get("vector", {}).get("expected_rank") is not None
            hybrid_hit = mode_entries.get("hybrid", {}).get("expected_rank") is not None
            agentic_hit = mode_entries.get("agentic", {}).get("expected_rank") is not None
            if not any([vector_hit, hybrid_hit, agentic_hit]):
                misses_by_kind[query["kind"]].append(query["question"])
            if not vector_hit and (hybrid_hit or agentic_hit):
                hybrid_only_recovery += 1
            if not vector_hit and expected:
                vector_misses += 1

        if misses_by_kind.get("filename"):
            lines.append(
                f"[召回失败] {len(misses_by_kind['filename'])} 个 filename 探针被所有模式漏掉"
                "（manifest 可能陈旧或 _normalize_source_path 丢了内容，检查 manifest['files']）。"
            )
        if misses_by_kind.get("symbol"):
            lines.append(
                f"[召回失败] {len(misses_by_kind['symbol'])} 个 symbol 探针被所有模式漏掉"
                "（symbol_index.jsonl 可能为空/陈旧，下次 build 会重建 AST 符号索引）。"
            )
        if user_probes_with_expected and hybrid_only_recovery >= max(1, user_probes_with_expected // 2):
            lines.append(
                f"[检索模式] {hybrid_only_recovery}/{user_probes_with_expected} 个探针向量模式漏掉但"
                " hybrid/agentic 找回，把默认 SEARCH_MODE 改成 hybrid（.env 里的 SEARCH_MODE）。"
            )

        # 规则 4：片段质量按 match_kind 分桶
        by_kind: Counter[tuple[str, str]] = Counter()
        total_hits_seen = 0
        for query in probes["queries"]:
            for mode_entry in query["modes"].values():
                for hit in mode_entry.get("top") or []:
                    total_hits_seen += 1
                    for flag in hit.get("flags") or []:
                        by_kind[(hit["match_kind"], flag)] += 1
        if total_hits_seen:
            flagged = sum(by_kind.values())
            if flagged / total_hits_seen > 0.3:
                top_offender = by_kind.most_common(1)[0] if by_kind else None
                if top_offender:
                    (kind_name, flag_name), count = top_offender
                    lines.append(
                        f"[片段质量] {flagged}/{total_hits_seen} 命中片段有问题（{flagged/total_hits_seen:.0%}），"
                        f"最主要的是 match_kind={kind_name} 触发 {flag_name} {count} 次。"
                        " 检查 ragbot_chunking.py 里对应 chunker 的边界设置。"
                    )

        # 规则 5：重 artifact 投入产出比
        skip_semantic = manifest_info.get("build_flags", {}).get("skip_semantic")
        hybrid_recall_hits = probes["recall_raw"].get("hybrid", [0, 0])[0]
        agentic_recall_hits = probes["recall_raw"].get("agentic", [0, 0])[0]
        if (
            not skip_semantic
            and total_tokens > 100_000
            and hybrid_recall_hits >= agentic_recall_hits
        ):
            lines.append(
                f"[投入产出] 已花 {total_tokens:,} token 跑 semantic，但 hybrid 召回 {hybrid_recall_hits}"
                f" ≥ agentic {agentic_recall_hits}。下轮 build 可先 SKIP_SEMANTIC=1；"
                " tests/test_hybrid_search.py:347 已验证能优雅降级。"
            )

    if not lines:
        lines.append("暂无触发的 verdict 规则。如果你仍觉得搜索不理想，把具体查询写进 JSON 跑 --probes 再看一次。")

    return lines


# ─────────────────────────────────────────────────────────
# 渲染
# ─────────────────────────────────────────────────────────
def render_human(report: dict[str, Any]) -> str:
    out: list[str] = []
    audit = report["audit"]
    out.append(f"OpenCortex 诊断报告 — {report['persist_dir']}")
    out.append("=" * 72)

    out.append("\n[artifact 产物]")
    for item in audit["artifacts"]:
        mark = "✓" if item["exists"] else "✗"
        size_kb = item["size_bytes"] / 1024.0 if item["size_bytes"] else 0
        out.append(f"  {mark} {item['path']:<52s} {size_kb:>10.1f} KB")

    tokens = audit["tokens"]
    out.append("\n[semantic token 消耗]")
    if not tokens.get("cache_exists"):
        out.append("  semantic_extract_cache.json 不存在（可能还没跑过 semantic 阶段或被清空）。")
    else:
        out.append(
            f"  api_calls={tokens['api_calls']:,}  prompt={tokens['prompt_tokens']:,}"
            f"  completion={tokens['completion_tokens']:,}  total={tokens['total_tokens']:,}"
        )
        out.append(
            f"  cached_sections={tokens['cached_sections']:,}  failed={tokens['failed_sections']:,}"
            f"  avg_per_call={tokens['avg_total_tokens_per_call']:.1f}"
        )
        out.append(
            f"  当前 prompt_version={tokens['current_prompt_version']}  陈旧条目={tokens['stale_entries']:,}"
        )
        if tokens["top_token_files"]:
            out.append("  token 消耗最大的文件：")
            for item in tokens["top_token_files"][:5]:
                out.append(
                    f"    {item['total_tokens']:>10,} tokens  sections={item['sections']:>3d}"
                    f"  {item['source']}"
                )

    manifest_info = audit["manifest"]
    out.append("\n[manifest 摘要]")
    out.append(
        f"  files={manifest_info['file_count']:,}  chunks={manifest_info['total_chunks']:,}"
        f"  chunks/file min/p50/p90/max="
        f"{manifest_info['chunks_per_file']['min']}/"
        f"{manifest_info['chunks_per_file']['p50']}/"
        f"{manifest_info['chunks_per_file']['p90']}/"
        f"{manifest_info['chunks_per_file']['max']}"
    )
    flags = manifest_info["build_flags"]
    out.append(
        f"  build_flags: skip_graph={flags['skip_graph']}  skip_semantic={flags['skip_semantic']}"
        f"  skip_wiki={flags['skip_wiki']}  SEARCH_MODE={manifest_info['search_mode_default']}"
    )
    if manifest_info["truncated_files"]:
        out.append(f"  截断文件数={len(manifest_info['truncated_files'])}（MAX_CHUNKS_PER_FILE 截断）")

    redflags = audit["chunking_redflags"]
    out.append("\n[chunking red flags]")
    for category, label in [
        ("many_tiny_chunks", "碎片过多"),
        ("single_chunk_large", "大文件单 chunk"),
        ("generic_fallback_with_known_extension", "已知扩展却走了 generic"),
    ]:
        entries = redflags.get(category) or []
        if entries:
            out.append(f"  {label}（{len(entries)} 个）：")
            for entry in entries[:5]:
                out.append(f"    - {entry}")

    if report.get("probes"):
        probes = report["probes"]
        out.append("\n[探针结果]")
        for query in probes["queries"]:
            summary_parts = []
            for mode, data in query["modes"].items():
                if data["expected_rank"] is None:
                    summary_parts.append(f"{mode}=miss")
                else:
                    summary_parts.append(
                        f"{mode}=#{data['expected_rank']}({data['found_via']})"
                    )
            expected = query.get("expected_source") or "-"
            out.append(
                f"  [{query['kind']:<8s}] q={query['question']!r:<30s}"
                f"  expected={expected}  {'  '.join(summary_parts)}"
            )
        out.append(f"\n  召回率：{probes['summary']['expected_recall']}")
        snippet = probes["summary"]["snippet_quality"]
        out.append(
            f"  片段问题：empty={snippet['empty']}  short={snippet['short']}"
            f"  no_query_token={snippet['no_query_token']}"
        )
    else:
        out.append("\n[探针结果] 已跳过（--skip-probes 或未提供 EMBED_API_KEY）")

    out.append("\n[verdict]")
    for line in report["verdict"]:
        out.append(f"  * {line}")

    return "\n".join(out)


# ─────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenCortex 索引诊断")
    parser.add_argument(
        "--persist-dir",
        default=os.getenv("CHROMA_PERSIST_DIR", ragbot.DEFAULT_FAISS_DIR),
    )
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--probes", type=str, default=None)
    parser.add_argument("--num-auto-probes", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=ragbot.DEFAULT_TOP_K)
    parser.add_argument("--modes", type=str, default="vector,hybrid,agentic")
    parser.add_argument("--allow-llm", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", dest="json_output", action="store_true")
    args = parser.parse_args(argv)

    persist_dir = Path(args.persist_dir).expanduser()
    if not persist_dir.exists():
        print(f"persist_dir 不存在：{persist_dir}", file=sys.stderr)
        return 0

    t0 = time.perf_counter()
    manifest = load_index_manifest(persist_dir) or {"files": []}

    report: dict[str, Any] = {
        "persist_dir": str(persist_dir),
        "audit": {
            "artifacts": audit_artifacts(persist_dir),
            "tokens": audit_tokens(persist_dir),
            "manifest": audit_manifest(manifest),
        },
    }

    chunk_kind_map: dict[str, list[str]] | None = None
    probes_result: dict[str, Any] | None = None

    if args.skip_probes:
        report["audit"]["chunking_redflags"] = audit_chunking_redflags(manifest, None)
    else:
        embed_api_key = os.getenv("EMBED_API_KEY", "").strip()
        if not embed_api_key:
            print(
                "[warn] 未设置 EMBED_API_KEY，跳过探针阶段；只产出静态审计。",
                file=sys.stderr,
            )
            report["audit"]["chunking_redflags"] = audit_chunking_redflags(manifest, None)
        else:
            from ragbot_runtime import load_search_bundle

            bundle = None
            try:
                bundle = load_search_bundle(
                    embed_api_key=embed_api_key,
                    embed_base_url=os.getenv("EMBED_BASE_URL") or None,
                    embed_model=os.getenv("EMBED_MODEL") or None,
                    persist_dir=persist_dir,
                )
            except Exception as exc:
                print(
                    f"[warn] load_search_bundle 失败：{exc}。很可能 index 产物不全，"
                    "常见原因是 fulltext_index.json / symbol_index.jsonl 缺失——"
                    "这本身就是搜索不理想的信号。跳过探针阶段。",
                    file=sys.stderr,
                )
                report["load_error"] = str(exc)

            if bundle is None:
                if "load_error" not in report:
                    print(
                        "[warn] load_search_bundle 返回 None（index.faiss 可能不存在），跳过探针。",
                        file=sys.stderr,
                    )
                report["audit"]["chunking_redflags"] = audit_chunking_redflags(manifest, None)
            else:
                chunk_kind_map = _collect_chunk_kinds(bundle)
                report["audit"]["chunking_redflags"] = audit_chunking_redflags(manifest, chunk_kind_map)

                requested_modes = [m.strip() for m in args.modes.split(",") if m.strip()]
                valid_modes = [m for m in requested_modes if m in ragbot.SEARCH_MODES]

                probes: list[dict[str, Any]] = []
                if args.probes:
                    probes.extend(load_user_probes(Path(args.probes).expanduser()))
                probes.extend(build_auto_probes(bundle, args.num_auto_probes, args.seed))

                if probes and valid_modes:
                    llm_env = {
                        "llm_api_key": os.getenv("LLM_API_KEY", "").strip(),
                        "llm_model": os.getenv("LLM_MODEL", "").strip(),
                        "llm_base_url": os.getenv("LLM_BASE_URL", "").strip(),
                    }
                    probes_result = run_probes(
                        bundle=bundle,
                        probes=probes,
                        modes=valid_modes,
                        top_k=args.top_k,
                        allow_llm=args.allow_llm,
                        llm_env=llm_env,
                    )
                    report["probes"] = probes_result

    report["verdict"] = build_verdict(report["audit"], probes_result)
    report["elapsed_seconds"] = round(time.perf_counter() - t0, 3)

    if args.json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(render_human(report))
        print(f"\n[elapsed] {report['elapsed_seconds']}s")
    return 0


def _collect_chunk_kinds(bundle: Any) -> dict[str, list[str]]:
    result: defaultdict[str, list[str]] = defaultdict(list)
    try:
        docstore = bundle.vectorstore.docstore._dict  # type: ignore[attr-defined]
    except AttributeError:
        return dict(result)
    for document in docstore.values():
        metadata = getattr(document, "metadata", None) or {}
        source = metadata.get("source") or ""
        kind = metadata.get("chunk_kind") or ""
        if source:
            result[source].append(kind)
    return dict(result)


if __name__ == "__main__":
    sys.exit(main())
