"""Aqua Voice / FluidVoice corpus manifest reader.

Reads the ``manifest.jsonl`` schema shared by Aqua Voice's harvester and
FluidVoice's ``DictationAudioHistoryStore.exportArchive``:

    {"audio": "audio/<file>.wav", "raw": "...", "corrected": "...",
     "duration": 6.8, "timestamp": "2026-07-07T00:34:38.487Z",
     "session_id": 125667720, "has_correction": true}

(Aqua Voice emits ``raw``/``corrected``; FluidVoice's ``AudioManifestRow`` emits
``rawTranscript``/``finalTranscript`` — :func:`read_aqua_manifest` accepts both.)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass
class AquaManifestEntry:
    """One corpus row. ``audio`` is resolved to an absolute path on read."""

    audio: Path
    raw: str
    corrected: str
    duration: float
    timestamp: str
    has_correction: bool
    session_id: int | None = None
    model: str | None = None  # FluidVoice rows carry the ASR model; Aqua's don't
    user_corrected: str | None = None  # FluidVoice edit-capture; strongest signal

    @property
    def is_genuine_correction(self) -> bool:
        """True iff the correction is more than whitespace.

        The Aqua corpus marks ``has_correction: true`` even when the only diff
        is a trailing space; the training signal is the *substantive* change, so
        callers should filter on this rather than on ``has_correction``.
        """
        return self.corrected.strip() != self.raw.strip() and bool(self.corrected.strip())

    @property
    def is_user_correction(self) -> bool:
        """True iff a manual user edit was captured (strongest training signal)."""
        uc = self.user_corrected.strip() if self.user_corrected else ""
        return bool(uc) and uc != self.raw.strip()


def _row_field(row: dict[str, Any], *names: str) -> str:
    """Read the first present key out of ``names`` (handles Aqua vs FluidVoice naming)."""
    for n in names:
        if n in row and row[n] is not None:
            return str(row[n])
    return ""


def read_aqua_manifest(path: Path) -> list[AquaManifestEntry]:
    """Read a manifest.jsonl, resolving ``audio`` to absolute paths.

    ``audio`` paths in the manifest are relative to the manifest's parent dir
    (e.g. ``"audio/AQ_x.wav"`` next to ``manifest.jsonl``).
    """
    base = path.parent
    entries: list[AquaManifestEntry] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            audio_rel = _row_field(row, "audio", "audio_filepath", "audioFilePath")
            audio = (base / audio_rel).resolve() if audio_rel else base
            raw = _row_field(row, "raw", "rawText", "raw_transcript", "rawTranscript")
            corrected = _row_field(row, "corrected", "final", "finalText",
                                   "final_transcript", "finalTranscript", "text", "content")
            duration = float(row.get("duration") or
                             (row.get("durationMilliseconds", 0) or 0) / 1000.0 or 0.0)
            timestamp = str(row.get("timestamp") or "")
            has_correction = bool(row.get("has_correction", corrected.strip() != raw.strip()))
            session_id = row.get("session_id") or row.get("sessionId")
            model = row.get("model")
            user_corrected = row.get("user_corrected")
            entries.append(AquaManifestEntry(
                audio=audio, raw=raw, corrected=corrected, duration=duration,
                timestamp=timestamp, has_correction=has_correction,
                session_id=int(session_id) if session_id is not None else None,
                model=str(model) if model else None,
                user_corrected=str(user_corrected) if user_corrected else None,
            ))
    return entries


# --------------------------------------------------------------------------- #
# Per-backend converters
# --------------------------------------------------------------------------- #
DEFAULT_INSTRUCTION = (
    "Clean up this speech-to-text transcript. Fix punctuation, capitalization, "
    "word choice, and formatting. Output only the corrected text."
)


def to_messages(entry: AquaManifestEntry, instruction: str = DEFAULT_INSTRUCTION) -> dict[str, Any]:
    """Build a ShareGPT/Gemma record: instruction+raw → corrected."""
    return {
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": entry.raw.strip()},
            {"role": "assistant", "content": entry.corrected.strip()},
        ]
    }


def to_nemo_entry(entry: AquaManifestEntry) -> dict[str, Any]:
    """Build a NeMo ASR manifest entry: the corrected text is the transcript target."""
    out: dict[str, Any] = {
        "audio_filepath": str(entry.audio),
        "text": entry.corrected.strip(),
    }
    if entry.duration > 0:
        out["duration"] = round(entry.duration, 3)
    return out


def iter_entries(entries: list[AquaManifestEntry]) -> Iterator[AquaManifestEntry]:
    for e in entries:
        yield e
