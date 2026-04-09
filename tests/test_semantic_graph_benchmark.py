from __future__ import annotations

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


class FakeMessage:
    def __init__(self, content: str):
        self.content = content
        self.usage_metadata = {"input_tokens": 24, "output_tokens": 10, "total_tokens": 34}


class SemanticBenchmarkLLM:
    def __init__(self, cases: list[dict[str, str]]):
        self.by_concept = {
            str(case["concept"]).lower(): {
                "concept": str(case["concept"]),
                "decision": str(case["decision"]),
            }
            for case in cases
        }

    def invoke(self, prompt: str) -> FakeMessage:
        lowered = prompt.lower()
        for concept_key, payload in self.by_concept.items():
            if concept_key in lowered:
                return FakeMessage(
                    json.dumps(
                        {
                            "concepts": [
                                {
                                    "name": payload["concept"],
                                    "summary": f"{payload['concept']} is a named workflow concept.",
                                    "aliases": [],
                                }
                            ],
                            "decisions": [
                                {
                                    "name": payload["decision"],
                                    "summary": f"{payload['decision']} is the chosen implementation policy.",
                                    "aliases": [],
                                    "rationale": [payload["concept"]],
                                }
                            ],
                        },
                        ensure_ascii=False,
                    )
                )
        return FakeMessage(json.dumps({"concepts": [], "decisions": []}, ensure_ascii=False))


def _semantic_benchmark_cases() -> list[dict[str, str]]:
    fixture_path = Path(__file__).parent / "fixtures" / "semantic_graph_benchmark_queries.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _code_fixture_for_case(case: dict[str, str]) -> str:
    function_name = Path(str(case["target_source"])).stem
    concept = str(case["concept"])
    decision = str(case["decision"])
    return f'''
def {function_name}(record_id: str) -> dict:
    """Run the {concept} workflow with {decision}."""
    return {{"record_id": record_id, "status": "ok"}}
'''.strip()


def _write_semantic_benchmark_corpus(root: Path, cases: list[dict[str, str]]) -> None:
    for case in cases:
        seed_path = root / str(case["seed_source"])
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        seed_path.write_text(
            "\n".join(
                [
                    f"# {case['id']}",
                    "",
                    f"{case['concept']} is the user-facing workflow name.",
                    f"The implementation uses {case['decision']} as the operating policy.",
                    "This note intentionally avoids file paths, imports, and code identifiers.",
                ]
            ),
            encoding="utf-8",
        )
        code_path = root / str(case["target_source"])
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(_code_fixture_for_case(case), encoding="utf-8")


def test_semantic_graph_benchmark_establishes_gain_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = _semantic_benchmark_cases()
    assert len(cases) == 10

    docs_dir = tmp_path / "docs"
    phase2a_dir = tmp_path / "phase2a"
    phase2b_dir = tmp_path / "phase2b"
    _write_semantic_benchmark_corpus(docs_dir, cases)

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    phase2a_bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-embed-key",
        persist_dir=str(phase2a_dir),
    )

    monkeypatch.setattr(
        ragbot,
        "make_llm",
        lambda *args, **kwargs: SemanticBenchmarkLLM(cases),
    )
    phase2b_bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-embed-key",
        persist_dir=str(phase2b_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://example.com/v1",
    )

    gain_case_ids: list[str] = []
    missing_targets: list[str] = []

    for case in cases:
        seed_source = str(case["seed_source"])
        target_source = str(case["target_source"])
        phase2a_targets = ragbot._expand_candidate_sources_detailed(
            phase2a_bundle,
            [seed_source],
            max_hops=1,
            max_extra_sources=12,
        ).sources
        phase2b_targets = ragbot._expand_candidate_sources_detailed(
            phase2b_bundle,
            [seed_source],
            max_hops=1,
            max_extra_sources=12,
        ).sources
        if target_source not in phase2b_targets:
            missing_targets.append(str(case["id"]))
        if target_source in phase2b_targets and target_source not in phase2a_targets:
            gain_case_ids.append(str(case["id"]))

    assert not missing_targets, f"semantic graph missed benchmark cases: {missing_targets}"
    assert len(gain_case_ids) >= 5, f"semantic graph gain cases: {gain_case_ids}"
