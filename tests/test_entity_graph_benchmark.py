from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

import ragbot


class FakeEmbeddings:
    def __init__(self, *_args, **_kwargs):
        self.dim = 48

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = text.lower().replace("\n", " ").split()
        if not tokens:
            return vector
        for token in tokens:
            slot = hash(token) % self.dim
            vector[slot] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


def _benchmark_cases() -> list[dict[str, str]]:
    fixture_path = Path(__file__).parent / "fixtures" / "entity_graph_benchmark_queries.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _code_fixture_for_target(target_source: str, symbol: str) -> str:
    templates = {
        "工程/bootstrap_session.py": """
def bootstrap_session(user_id: str) -> dict:
    return {"user_id": user_id, "status": "ready"}
""".strip(),
        "工程/render_dashboard.py": """
def render_dashboard(account_id: str) -> dict:
    return {"account_id": account_id, "panel": "overview"}
""".strip(),
        "工程/reconcile_invoice.py": """
def reconcile_invoice(invoice_id: str) -> dict:
    return {"invoice_id": invoice_id, "status": "matched"}
""".strip(),
        "工程/archive_report.py": """
def archive_report(report_id: str) -> dict:
    return {"report_id": report_id, "state": "archived"}
""".strip(),
    }
    text = templates.get(target_source)
    if text is None:
        raise AssertionError(f"missing code fixture for {target_source} / {symbol}")
    return text


def _write_entity_benchmark_corpus(root: Path, cases: list[dict[str, str]]) -> None:
    code_targets: dict[str, str] = {}
    for case in cases:
        code_targets[case["target_source"]] = case["symbol"]
        seed_path = root / case["seed_source"]
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        seed_path.write_text(
            "\n".join(
                [
                    f"# {case['id']}",
                    "",
                    f"本说明记录了 `{case['symbol']}()` 的业务语义。",
                    f"排查 {case['question']}",
                    "这里只描述职责，不直接给出源码路径。",
                ]
            ),
            encoding="utf-8",
        )

    for target_source, symbol in code_targets.items():
        code_path = root / target_source
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(_code_fixture_for_target(target_source, symbol), encoding="utf-8")


def test_entity_graph_benchmark_establishes_gain_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cases = _benchmark_cases()
    assert len(cases) == 20

    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_entity_benchmark_corpus(docs_dir, cases)

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())

    bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-key",
        persist_dir=str(index_dir),
    )
    doc_only_bundle = replace(
        bundle,
        entity_graph={"version": 1, "node_count": 0, "edge_count": 0, "nodes": [], "edges": []},
        entity_nodes_by_id={},
        entity_edges_by_source={},
    )

    gained_case_ids: list[str] = []
    missing_entity_targets: list[str] = []

    for case in cases:
        seed_source = case["seed_source"]
        expected_target = case["target_source"]
        doc_targets = ragbot._expand_candidate_sources_detailed(
            doc_only_bundle,
            [seed_source],
            max_hops=1,
            max_extra_sources=12,
        ).sources
        entity_expansion = ragbot._expand_candidate_sources_detailed(
            bundle,
            [seed_source],
            max_hops=1,
            max_extra_sources=12,
        )
        entity_targets = entity_expansion.sources
        assert entity_expansion.strategy == "entity_graph"

        if expected_target not in entity_targets:
            missing_entity_targets.append(case["id"])
        if expected_target in entity_targets and expected_target not in doc_targets:
            gained_case_ids.append(case["id"])

    assert not missing_entity_targets, f"entity graph missed benchmark cases: {missing_entity_targets}"
    assert len(gained_case_ids) >= 5, f"entity graph gain cases: {gained_case_ids}"
