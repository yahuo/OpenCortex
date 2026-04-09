from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ragbot import QueryPlan, SearchBundle, SearchHit


def _core():
    import ragbot as core

    return core


@lru_cache(maxsize=2048)
def _read_cached_text(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8", errors="ignore")


def _search_mode(value: str | None) -> str:
    core = _core()
    mode = (value or core.os.getenv("SEARCH_MODE", core.DEFAULT_SEARCH_MODE)).strip().lower()
    return mode if mode in core.SEARCH_MODES else core.DEFAULT_SEARCH_MODE


def _search_max_steps() -> int:
    core = _core()
    raw = core.os.getenv("SEARCH_MAX_STEPS", str(core.SEARCH_MAX_STEPS_DEFAULT)).strip()
    try:
        value = int(raw)
    except ValueError:
        return core.SEARCH_MAX_STEPS_DEFAULT
    return 1 if value <= 1 else 2


def _bundle_sources(
    bundle: SearchBundle,
    kb: str | None = None,
    allowed_sources: set[str] | None = None,
) -> list[str]:
    results: list[str] = []
    for entry in bundle.files:
        source = entry.get("name", "")
        if not source:
            continue
        if kb is not None and entry.get("kb") != kb:
            continue
        if allowed_sources is not None and source not in allowed_sources:
            continue
        results.append(source)
    return results


def _extract_query_plan(question: str) -> QueryPlan:
    core = _core()
    quoted = re.findall(r"`([^`]+)`", question)
    quoted += re.findall(r'"([^"]+)"', question)
    quoted += re.findall(r"“([^”]+)”", question)
    quoted += re.findall(r"'([^']+)'", question)
    quoted_clean_map: dict[str, str] = {}
    for item in quoted:
        clean = item.strip().strip(".,:;!?()[]{}")
        if clean:
            quoted_clean_map[clean.lower()] = clean

    raw_tokens = (
        quoted
        + core.TOKEN_RE.findall(question)
        + core.NON_ASCII_FILENAME_RE.findall(question)
    )
    path_globs: list[str] = []
    symbols: list[str] = []
    keywords: list[str] = []

    for token in raw_tokens:
        clean = token.strip().strip(".,:;!?()[]{}")
        if not clean:
            continue
        has_supported_suffix = any(
            clean.lower().endswith(suffix) and len(clean) > len(suffix)
            for suffix in core.SUPPORTED_TEXT_SUFFIXES
        )
        if clean.lower() not in core._EXTENSION_ALIASES:
            keywords.append(clean)

        if core.PATHISH_RE.fullmatch(clean) or has_supported_suffix or "/" in clean or "*" in clean:
            if clean.startswith(".") and clean.count(".") == 1 and "/" not in clean:
                path_globs.append(f"*{clean}")
            elif "*" in clean:
                path_globs.append(clean)
            else:
                path_globs.append(f"*{clean}*")

        if clean.startswith(".") and clean.count(".") == 1 and len(clean) <= 8:
            path_globs.append(f"*{clean}")

        stripped_call = clean[:-2] if clean.endswith("()") else clean
        quoted_token = stripped_call.lower() in quoted_clean_map or clean.lower() in quoted_clean_map
        if core.SYMBOL_RE.fullmatch(stripped_call) and (
            quoted_token or "_" in stripped_call or any(char.isupper() for char in stripped_call)
        ):
            symbols.append(stripped_call)

        if quoted_token and "." in clean and all(core.SYMBOL_RE.fullmatch(part) for part in clean.split(".")):
            symbols.append(clean)

        if not has_supported_suffix and "." in stripped_call and all(part for part in stripped_call.split(".")):
            tail = stripped_call.split(".")[-1]
            if core.SYMBOL_RE.fullmatch(tail):
                symbols.append(tail)

    for match in core.CALL_RE.findall(question):
        symbols.append(match)
        keywords.append(f"{match}(")

    lowered_question = question.lower()
    for alias, pattern in core._EXTENSION_ALIASES.items():
        if re.search(rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])", lowered_question):
            path_globs.append(pattern)

    return core.QueryPlan(
        symbols=core._dedupe_strings(symbols),
        keywords=core._dedupe_strings(keywords),
        path_globs=core._dedupe_strings(path_globs),
        semantic_query=question.strip(),
        reason="rule planner",
    )


def _hit_priority(match_kind: str) -> int:
    order = {"ast": 4, "grep": 3, "vector": 2, "glob": 1}
    return order.get(match_kind, 0)


def _merge_primary_kind(current: SearchHit, candidate: SearchHit) -> None:
    if _hit_priority(candidate.match_kind) > _hit_priority(current.match_kind):
        current.match_kind = candidate.match_kind
        current.metadata.update(candidate.metadata)
        if candidate.snippet:
            current.snippet = candidate.snippet
        if candidate.line_start is not None:
            current.line_start = candidate.line_start
        if candidate.line_end is not None:
            current.line_end = candidate.line_end


def _finalize_hits(
    grouped_hits: dict[str, list[SearchHit]],
    top_k: int,
) -> list[SearchHit]:
    core = _core()
    merged: dict[tuple[str, int | None, int | None, str], SearchHit] = {}
    for match_kind, hits in grouped_hits.items():
        weight = core.RRF_WEIGHTS[match_kind]
        for rank, hit in enumerate(hits, start=1):
            key = hit.dedupe_key()
            fused = merged.get(key)
            if fused is None:
                fused = core.SearchHit(
                    source=hit.source,
                    match_kind=hit.match_kind,
                    snippet=hit.snippet,
                    score=0.0,
                    line_start=hit.line_start,
                    line_end=hit.line_end,
                    metadata=dict(hit.metadata),
                )
                merged[key] = fused

            fused.score += weight / (core.RRF_K + rank)
            if hit.metadata.get("exact_symbol"):
                fused.score += 0.25
            if hit.metadata.get("exact_path"):
                fused.score += 0.15
            _merge_primary_kind(fused, hit)

    ranked = sorted(
        merged.values(),
        key=lambda item: (item.score, _hit_priority(item.match_kind), item.source),
        reverse=True,
    )

    final_hits: list[SearchHit] = []
    per_file_count: dict[str, int] = {}
    for hit in ranked:
        if per_file_count.get(hit.source, 0) >= 2:
            continue
        final_hits.append(hit)
        per_file_count[hit.source] = per_file_count.get(hit.source, 0) + 1
        if len(final_hits) >= top_k:
            break
    return final_hits


def _first_non_empty_lines(cache_path: Path | None, limit: int = 3) -> str:
    core = _core()
    if cache_path is None or not cache_path.exists():
        return ""
    lines = [
        line.strip()
        for line in core._read_cached_text(str(cache_path)).splitlines()
        if line.strip()
    ]
    return "\n".join(lines[:limit])


def glob_search(
    bundle: SearchBundle,
    path_globs: list[str],
    kb: str | None = None,
    allowed_sources: set[str] | None = None,
) -> list[SearchHit]:
    core = _core()
    if not path_globs:
        return []

    results: list[SearchHit] = []
    for source in core._bundle_sources(bundle, kb=kb, allowed_sources=allowed_sources):
        path_lower = source.lower()
        basename_lower = Path(source).name.lower()
        matched = False
        exact_path = False
        for pattern in path_globs:
            normalized_pattern = pattern.lower()
            if core.fnmatch.fnmatch(path_lower, normalized_pattern) or core.fnmatch.fnmatch(
                basename_lower, normalized_pattern
            ):
                matched = True
                stripped = normalized_pattern.strip("*")
                exact_path = stripped == path_lower or stripped == basename_lower
                break
            stripped = normalized_pattern.strip("*")
            if stripped and (stripped in path_lower or stripped in basename_lower):
                matched = True
                exact_path = stripped == basename_lower or stripped == path_lower
                break
        if not matched:
            continue

        cache_path = bundle.cache_path_for(source)
        snippet = _first_non_empty_lines(cache_path) or source
        results.append(
            core.SearchHit(
                source=source,
                match_kind="glob",
                snippet=snippet,
                score=1.0,
                metadata={"exact_path": exact_path},
            )
        )

    return results


def _keyword_score(line: str, keyword: str) -> float:
    core = _core()
    lowered_line = line.lower()
    lowered_keyword = keyword.lower()
    if lowered_keyword not in lowered_line:
        return 0.0
    if line == keyword:
        return 1.2
    if core.WORD_CHARS_RE.fullmatch(keyword):
        if re.search(rf"\b{re.escape(keyword)}\b", line, re.IGNORECASE):
            return 1.0
    return 0.75


def _grep_hits_from_matches(
    bundle: SearchBundle,
    matches: dict[tuple[str, int], dict[str, Any]],
) -> list[SearchHit]:
    core = _core()
    hits: list[SearchHit] = []
    for (source, line_no), payload in matches.items():
        cache_path = bundle.cache_path_for(source)
        if cache_path is None or not cache_path.exists():
            continue
        lines = core._read_cached_text(str(cache_path)).splitlines()
        if not lines:
            continue
        start = max(1, line_no - 4)
        end = min(len(lines), line_no + 4)
        snippet = "\n".join(lines[start - 1 : end]).strip()
        exact_text = any(score >= 1.0 for score in payload["scores"])
        hits.append(
            core.SearchHit(
                source=source,
                match_kind="grep",
                snippet=snippet,
                score=max(payload["scores"]) + len(payload["keywords"]) * 0.05,
                line_start=line_no,
                line_end=line_no,
                metadata={
                    "matched_keywords": sorted(payload["keywords"]),
                    "exact_symbol": exact_text,
                },
            )
        )
    return sorted(hits, key=lambda item: (item.score, -int(item.line_start or 0)), reverse=True)


def _grep_search_with_rg(
    bundle: SearchBundle,
    keywords: list[str],
    sources: list[str],
) -> list[SearchHit]:
    core = _core()
    rg_path = core.shutil.which("rg")
    if rg_path is None or not keywords or not sources:
        return []

    source_to_cache: dict[str, str] = {}
    for source in sources:
        cache_path = bundle.cache_path_for(source)
        if cache_path is None or not cache_path.exists():
            continue
        source_to_cache[source] = str(cache_path)
    if not source_to_cache:
        return []

    cache_to_source = {cache: source for source, cache in source_to_cache.items()}
    matches: dict[tuple[str, int], dict[str, Any]] = {}
    cmd = [
        rg_path,
        "-n",
        "-i",
        "--fixed-strings",
        "--no-heading",
        "--color=never",
    ]
    for keyword in keywords:
        cmd.extend(["-e", keyword])
    cmd.extend(source_to_cache.values())

    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    if completed.returncode not in (0, 1):
        return []

    for line in completed.stdout.splitlines():
        try:
            path_str, line_number_str, content = line.split(":", 2)
            source = cache_to_source.get(path_str)
            if source is None:
                continue
            line_number = int(line_number_str)
        except ValueError:
            continue

        matched_keywords: list[str] = []
        scores: list[float] = []
        for keyword in keywords:
            score = _keyword_score(content, keyword)
            if score <= 0:
                continue
            matched_keywords.append(keyword)
            scores.append(score)
        if not scores:
            continue

        key = (source, line_number)
        bucket = matches.setdefault(key, {"keywords": set(), "scores": []})
        bucket["keywords"].update(matched_keywords)
        bucket["scores"].extend(scores)

    return _grep_hits_from_matches(bundle, matches)


def _grep_search_python(
    bundle: SearchBundle,
    keywords: list[str],
    sources: list[str],
) -> list[SearchHit]:
    core = _core()
    matches: dict[tuple[str, int], dict[str, Any]] = {}
    for source in sources:
        cache_path = bundle.cache_path_for(source)
        if cache_path is None or not cache_path.exists():
            continue
        lines = core._read_cached_text(str(cache_path)).splitlines()
        for line_no, line in enumerate(lines, start=1):
            scores = []
            matched_keywords = []
            for keyword in keywords:
                score = _keyword_score(line, keyword)
                if score > 0:
                    scores.append(score)
                    matched_keywords.append(keyword)
            if not scores:
                continue
            key = (source, line_no)
            bucket = matches.setdefault(key, {"keywords": set(), "scores": []})
            bucket["keywords"].update(matched_keywords)
            bucket["scores"].extend(scores)

    return _grep_hits_from_matches(bundle, matches)


def grep_search(
    bundle: SearchBundle,
    keywords: list[str],
    kb: str | None = None,
    allowed_sources: set[str] | None = None,
) -> list[SearchHit]:
    core = _core()
    keywords = core._dedupe_strings(keywords)
    if not keywords:
        return []

    sources = core._bundle_sources(bundle, kb=kb, allowed_sources=allowed_sources)
    hits = _grep_search_with_rg(bundle, keywords, sources)
    if hits:
        return hits
    return _grep_search_python(bundle, keywords, sources)


def _symbol_match_score(record: dict[str, Any], query_terms: list[str]) -> tuple[float, bool]:
    exact_symbol = False
    best = 0.0
    names = {
        record.get("name", "").lower(),
        record.get("qualified_name", "").lower(),
    }
    for term in query_terms:
        lowered = term.lower().rstrip("()")
        if not lowered:
            continue
        if lowered in names:
            exact_symbol = True
            best = max(best, 1.5)
            continue
        if any(lowered == name.split(".")[-1] for name in names if name):
            exact_symbol = True
            best = max(best, 1.4)
            continue
        if any(lowered in name for name in names if name):
            best = max(best, 0.9)
    return best, exact_symbol


def ast_search(
    bundle: SearchBundle,
    query_plan: QueryPlan,
    kb: str | None = None,
    allowed_sources: set[str] | None = None,
) -> list[SearchHit]:
    core = _core()
    if not bundle.symbol_index:
        return []

    query_terms = core._dedupe_strings([*query_plan.symbols, *query_plan.keywords])
    if not query_terms:
        return []

    valid_sources = set(core._bundle_sources(bundle, kb=kb, allowed_sources=allowed_sources))
    hits: list[SearchHit] = []
    for record in bundle.symbol_index:
        source = record.get("source", "")
        if valid_sources and source not in valid_sources:
            continue

        score, exact_symbol = _symbol_match_score(record, query_terms)
        if score <= 0:
            continue

        cache_path = bundle.cache_path_for(source)
        if cache_path is None or not cache_path.exists():
            continue
        lines = core._read_cached_text(str(cache_path)).splitlines()
        line_start = int(record.get("line_start") or 1)
        line_end = int(record.get("line_end") or line_start)
        capped_end = min(line_end, line_start + 119, len(lines))
        snippet = "\n".join(lines[line_start - 1 : capped_end]).strip()
        if not snippet:
            continue
        hits.append(
            core.SearchHit(
                source=source,
                match_kind="ast",
                snippet=snippet,
                score=score,
                line_start=line_start,
                line_end=capped_end,
                metadata={
                    "kind": record.get("kind", ""),
                    "signature": record.get("signature", ""),
                    "exact_symbol": exact_symbol,
                },
            )
        )

    return sorted(hits, key=lambda item: (item.score, -int(item.line_start or 0)), reverse=True)

