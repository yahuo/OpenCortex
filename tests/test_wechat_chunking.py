from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

import ragbot
from ragbot_chunking import _iter_wechat_markdown_chunks_from_file


def _msg(sender: str, ts: datetime | None, content: str, line: int) -> dict:
    return {
        "sender": sender,
        "timestamp": ts,
        "content": content,
        "line_start": line,
        "line_end": line,
    }


def test_merge_small_wechat_chunks_collapses_tiny_windows():
    base = datetime(2026, 4, 1, 10, 0, 0)
    chunks = [
        [_msg("A", base, "x" * 100, 1)],
        [_msg("B", base + timedelta(minutes=2), "y" * 100, 2)],
        [_msg("C", base + timedelta(minutes=4), "z" * 100, 3)],
    ]
    merged = ragbot._merge_small_wechat_chunks(
        chunks, min_chars=800, max_chars=3000, quiet_gap_minutes=120
    )
    assert len(merged) == 1
    assert [m["sender"] for m in merged[0]] == ["A", "B", "C"]


def test_merge_respects_quiet_gap():
    base = datetime(2026, 4, 1, 9, 0, 0)
    chunks = [
        [_msg("A", base, "x" * 200, 1)],
        [_msg("B", base + timedelta(hours=4), "y" * 200, 2)],
    ]
    merged = ragbot._merge_small_wechat_chunks(
        chunks, min_chars=800, max_chars=3000, quiet_gap_minutes=120
    )
    assert len(merged) == 2


def test_merge_preserves_large_windows():
    base = datetime(2026, 4, 1, 8, 0, 0)
    big = [_msg("A", base, "x" * 2000, 1)]
    next_big = [_msg("B", base + timedelta(minutes=5), "y" * 1500, 2)]
    merged = ragbot._merge_small_wechat_chunks(
        [big, next_big], min_chars=800, max_chars=3000, quiet_gap_minutes=120
    )
    assert len(merged) == 2


def test_merge_absorbs_small_into_adjacent_large():
    base = datetime(2026, 4, 1, 8, 0, 0)
    big = [_msg("A", base, "x" * 1500, 1)]
    small = [_msg("B", base + timedelta(minutes=5), "y" * 100, 2)]
    merged = ragbot._merge_small_wechat_chunks(
        [big, small], min_chars=800, max_chars=3000, quiet_gap_minutes=120
    )
    assert len(merged) == 1
    assert [m["sender"] for m in merged[0]] == ["A", "B"]


def test_merge_stops_when_max_exceeded():
    base = datetime(2026, 4, 1, 8, 0, 0)
    a = [_msg("A", base, "x" * 1500, 1)]
    b = [_msg("B", base + timedelta(minutes=5), "y" * 700, 2)]
    c = [_msg("C", base + timedelta(minutes=10), "z" * 1500, 3)]
    merged = ragbot._merge_small_wechat_chunks(
        [a, b, c], min_chars=800, max_chars=3000, quiet_gap_minutes=120
    )
    # a+b = 2200 ≤ 3000 → merge; (a+b)+c = 3700 > 3000 → c stays separate
    assert len(merged) == 2
    assert [m["sender"] for m in merged[0]] == ["A", "B"]
    assert [m["sender"] for m in merged[1]] == ["C"]


def test_chunk_by_time_window_applies_merge():
    base = datetime(2026, 4, 1, 7, 0, 0)
    messages = []
    for i in range(6):
        messages.append(_msg(f"U{i}", base + timedelta(minutes=i * 35), "hi" * 30, i + 1))
    chunks = ragbot._chunk_by_time_window(messages, window_minutes=30)
    # Each message would be its own 30-min window; merge collapses them since each is tiny
    assert len(chunks) < 6


def _build_wechat_file(path: Path, messages: list[tuple[str, datetime, str]]) -> None:
    lines: list[str] = []
    for sender, ts, body in messages:
        lines.append(f"**{sender}** `[{ts.strftime('%Y-%m-%d %H:%M:%S')}]`")
        lines.append(body)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_streaming_path_emits_merged_chunks(tmp_path: Path):
    base = datetime(2026, 4, 1, 9, 0, 0)
    messages = [
        ("A", base + timedelta(minutes=i * 5), f"message body {i} " * 5)
        for i in range(20)
    ]
    file = tmp_path / "chat.md"
    _build_wechat_file(file, messages)

    chunks = list(_iter_wechat_markdown_chunks_from_file(file, window_minutes=30))
    # Without merge each tiny window would create many chunks; merge should collapse to a handful.
    assert 0 < len(chunks) < 8
    for chunk in chunks:
        assert chunk.text


def test_streaming_path_respects_quiet_gap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WECHAT_QUIET_GAP_MINUTES", "60")
    base = datetime(2026, 4, 1, 9, 0, 0)
    early = [("A", base + timedelta(minutes=i * 5), "early " * 4) for i in range(3)]
    late = [("B", base + timedelta(hours=5) + timedelta(minutes=i * 5), "late " * 4) for i in range(3)]
    file = tmp_path / "chat.md"
    _build_wechat_file(file, early + late)

    chunks = list(_iter_wechat_markdown_chunks_from_file(file, window_minutes=30))
    # Quiet gap (5h > 60min) prevents early + late from merging into a single chunk.
    assert len(chunks) >= 2
    joined = "\n".join(chunk.text for chunk in chunks)
    assert "early" in joined and "late" in joined


def test_max_chunks_per_file_still_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAX_CHUNKS_PER_FILE", "2")
    monkeypatch.setenv("WECHAT_MIN_CHUNK_CHARS", "1")  # disable merging by setting threshold low
    monkeypatch.setenv("WECHAT_MAX_CHUNK_CHARS", "10")  # tiny cap → cannot merge
    base = datetime(2026, 4, 1, 9, 0, 0)
    messages = [
        ("A", base + timedelta(hours=i), f"body{i} " * 3)
        for i in range(8)
    ]
    file = tmp_path / "chat.md"
    _build_wechat_file(file, messages)

    from ragbot_chunking import _iter_chunks_from_cached_file

    chunks = list(_iter_chunks_from_cached_file(file, ".md", "wechat_markdown"))
    assert len(chunks) <= 2
