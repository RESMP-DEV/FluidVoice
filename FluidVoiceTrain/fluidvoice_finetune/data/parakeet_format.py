"""Parakeet NeMo manifest converter.

NVIDIA NeMo ASR training expects a JSONL *manifest*: one JSON object per audio
file, each with at minimum:

    {"audio_filepath": "/abs/path.wav", "text": "transcript", "duration": 2.43}

This module converts (audio_path, transcript[, duration]) tuples into that
shape and writes a manifest. Duration is computed via the ``soundfile`` /
``librosa`` dependency if available; if missing, the caller must supply it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


def _compute_duration(audio_path: Path) -> float | None:
    """Best-effort duration in seconds. Returns None if audio libs are absent
    or the file can't be read."""
    try:
        import soundfile as sf  # type: ignore[import]
    except ImportError:
        return None
    try:
        info = sf.info(str(audio_path))
        return float(info.frames / info.samplerate) if info.samplerate else None
    except Exception:
        return None


def to_manifest_entry(audio_filepath: Path | str, text: str, duration: float | None = None) -> dict[str, Any]:
    """Build a single NeMo manifest entry."""
    text = text.strip()
    if not text:
        raise ValueError("text must be non-empty")
    audio_path = Path(audio_filepath)
    if duration is None:
        duration = _compute_duration(audio_path)
    entry: dict[str, Any] = {
        "audio_filepath": str(audio_path),
        "text": text,
    }
    if duration is not None:
        entry["duration"] = round(float(duration), 3)
    return entry


def write_manifest(entries: Iterable[dict[str, Any]], out_path: Path) -> int:
    """Write a NeMo JSONL manifest. Returns entry count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            if "audio_filepath" not in entry or "text" not in entry:
                raise ValueError(f"manifest entry missing required keys: {entry}")
            fh.write(json.dumps(entry, ensure_ascii=False))
            fh.write("\n")
            count += 1
    return count


def read_manifest(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def total_duration_hours(entries: Iterable[dict[str, Any]]) -> float:
    """Sum durations across entries; entries without duration contribute 0."""
    total = 0.0
    for e in entries:
        d = e.get("duration")
        if isinstance(d, (int, float)) and math.isfinite(d):
            total += float(d)
    return total / 3600.0
