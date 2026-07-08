"""Format-agnostic dataset ingest → train/val split → backend format.

The input dataset can be one of several shapes; :func:`prepare_dataset`
normalizes it into the target backend's format (``messages`` for Gemma,
``nemo-manifest`` for Parakeet) and writes ``train.jsonl`` + ``val.jsonl``.

Recognized input shapes (auto-detected unless ``--dataset-format`` forces one):

- **messages**: JSONL of ``{"messages": [...]}`` or ``{"raw":..., "target":...}``
  → emitted as Gemma messages JSONL.
- **nemo-manifest**: NeMo JSONL ``{"audio_filepath","text"[,"duration"]}``
  → emitted unchanged (split only).
- **raw pairs CSV/JSONL**: ``raw,target`` CSV or ``{"raw":..,"target":..}`` JSONL
  → Gemma messages JSONL.

Once the dev provides the real dataset, add its specific loader here.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Iterator

from ..config import DatasetFormat
from .aqua import (
    AquaManifestEntry,
    read_aqua_manifest,
    to_messages as aqua_to_messages,
    to_nemo_entry as aqua_to_nemo,
)
from .gemma_format import to_messages, write_messages_jsonl
from .parakeet_format import write_manifest


# --------------------------------------------------------------------------- #
# Ingest: raw input → list of normalized dicts
# --------------------------------------------------------------------------- #
def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _detect_format(path: Path, explicit: DatasetFormat) -> DatasetFormat:
    if explicit is not DatasetFormat.AUTO:
        return explicit
    name = path.name.lower()
    if name.endswith(".csv"):
        return DatasetFormat.RAW
    # Peek first JSONL record.
    try:
        first = next(_iter_jsonl(path))
    except (StopIteration, FileNotFoundError, json.JSONDecodeError):
        return DatasetFormat.MESSAGES
    if {"audio_filepath", "text"} <= set(first):
        return DatasetFormat.NEMO_MANIFEST
    if "messages" in first:
        return DatasetFormat.MESSAGES
    if {"raw", "target"} <= set(first):
        return DatasetFormat.RAW
    return DatasetFormat.MESSAGES


def _load_records(path: Path, source_fmt: DatasetFormat) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if source_fmt is DatasetFormat.NEMO_MANIFEST:
        records = list(_iter_jsonl(path))
    elif source_fmt is DatasetFormat.MESSAGES:
        records = list(_iter_jsonl(path))
    elif source_fmt is DatasetFormat.RAW:
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as fh:
                # Sniff for a header: if first row is literally "raw,target", treat
                # it as a header (DictReader auto-detects). Otherwise force the
                # fieldnames so headerless two-column CSVs still parse.
                first = fh.readline()
                fh.seek(0)
                has_header = first.strip().lower() in {"raw,target", "input,output", "prompt,response"}
                reader = csv.DictReader(fh) if has_header else csv.DictReader(fh, fieldnames=["raw", "target"])
                for row in reader:
                    if row.get("raw") and row.get("target"):
                        records.append({"raw": row["raw"], "target": row["target"]})
        else:
            records = list(_iter_jsonl(path))
    else:  # RAW JSONL with raw/target
        records = list(_iter_jsonl(path))
    return records


# --------------------------------------------------------------------------- #
# Convert: normalized records → target backend format records
# --------------------------------------------------------------------------- #
def _to_target_records(records: list[dict[str, Any]], target_fmt: DatasetFormat) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in records:
        if target_fmt is DatasetFormat.NEMO_MANIFEST:
            if {"audio_filepath", "text"} <= set(rec):
                out.append(rec)
            else:
                raise ValueError(
                    f"Cannot convert record to nemo-manifest (needs audio_filepath+text): {rec}"
                )
        else:  # messages
            if "messages" in rec:
                out.append(rec)
            elif {"raw", "target"} <= set(rec):
                out.append(to_messages(rec["raw"], rec["target"]))
            else:
                raise ValueError(f"Cannot convert record to messages: {rec}")
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def prepare_dataset(
    dataset: Path,
    out_dir: Path,
    target_fmt: DatasetFormat,
    source_fmt: DatasetFormat = DatasetFormat.AUTO,
    val_split: float = 0.05,
    seed: int = 42,
) -> int:
    """Ingest ``dataset``, split, and write ``train.jsonl`` + ``val.jsonl``.

    Returns total record count written (train + val).
    """
    detected = _detect_format(dataset, source_fmt)
    records = _load_records(dataset, detected)
    if not records:
        raise ValueError(f"No records loaded from {dataset}")

    target_records = _to_target_records(records, target_fmt)

    rng = random.Random(seed)
    rng.shuffle(target_records)
    n_val = int(len(target_records) * val_split)
    val = target_records[:n_val]
    train = target_records[n_val:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    if target_fmt is DatasetFormat.NEMO_MANIFEST:
        write_manifest(train, train_path)
        write_manifest(val, val_path)
    else:
        write_messages_jsonl(train, train_path)
        write_messages_jsonl(val, val_path)

    return len(target_records)


# --------------------------------------------------------------------------- #
# Aqua / FluidVoice corpus preparation
# --------------------------------------------------------------------------- #
def _normalize_text(s: str) -> str:
    """Strip + collapse internal whitespace. Preserves punctuation (that's the signal)."""
    import re
    return re.sub(r"[ \t]+", " ", s.strip())


def _audio_exists(entry: AquaManifestEntry) -> bool:
    return entry.audio.exists() and entry.audio.is_file()


def prepare_aqua_corpus(
    src_dir: Path,
    out_dir: Path,
    target_fmt: DatasetFormat,
    val_split: float = 0.1,
    seed: int = 42,
    *,
    require_genuine_correction: bool | None = None,
    require_audio: bool | None = None,
    watermark: str | None = None,
) -> dict[str, Any]:
    """Prepare an Aqua/FluidVoice corpus dir into a backend's train/val split.

    Reads ``<src_dir>/manifest.jsonl`` (and expects ``<src_dir>/audio/*.wav``).
    Writes ``<out_dir>/{train,val}.jsonl`` (+ ``<out_dir>/audio/`` for nemo).

    Args:
        require_genuine_correction: For Gemma (messages), drop rows whose only
            diff is whitespace (default True). For Parakeet (nemo-manifest),
            keep all rows with audio (default False) since ASR learns from
            every utterance regardless of whether the transcript was corrected.
        require_audio: For Parakeet, drop rows whose audio file is missing
            (default True). For Gemma, audio is irrelevant (default False).
        watermark: ISO timestamp; only entries with ``timestamp > watermark``
            are emitted (incremental training). None = all entries.

    Returns a summary dict (also written to ``<out_dir>/manifest_summary.json``):
        ``{"total": N, "kept": K, "filtered_no_correction": ..., "filtered_no_audio": ...,
           "filtered_by_watermark": ..., "train": T, "val": V, "audio_hours": H}``.
    """
    manifest_path = src_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.jsonl in {src_dir}")

    entries = read_aqua_manifest(manifest_path)
    total = len(entries)

    # Defaults differ by backend.
    if require_genuine_correction is None:
        require_genuine_correction = target_fmt is DatasetFormat.MESSAGES
    if require_audio is None:
        require_audio = target_fmt is DatasetFormat.NEMO_MANIFEST

    # Apply filters, tracking why each entry was dropped.
    kept: list[AquaManifestEntry] = []
    stats = {"total": total, "filtered_no_correction": 0, "filtered_no_audio": 0,
             "filtered_by_watermark": 0}
    for e in entries:
        if watermark and e.timestamp and e.timestamp <= watermark:
            stats["filtered_by_watermark"] += 1
            continue
        if require_genuine_correction and not e.is_genuine_correction:
            stats["filtered_no_correction"] += 1
            continue
        if require_audio and not _audio_exists(e):
            stats["filtered_no_audio"] += 1
            continue
        kept.append(e)

    # Normalize text fields in place.
    for e in kept:
        e.raw = _normalize_text(e.raw)
        e.corrected = _normalize_text(e.corrected)

    rng = random.Random(seed)
    rng.shuffle(kept)
    n_val = int(len(kept) * val_split)
    val_entries = kept[:n_val]
    train_entries = kept[n_val:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    audio_hours = 0.0
    if target_fmt is DatasetFormat.NEMO_MANIFEST:
        # Copy/symlink audio into out_dir/audio/ and rewrite paths absolute.
        out_audio = out_dir / "audio"
        out_audio.mkdir(parents=True, exist_ok=True)
        train_records: list[dict[str, Any]] = []
        for e in train_entries + val_entries:
            audio_hours += e.duration / 3600.0
            # Symlink to avoid duplicating (potentially large) WAVs.
            link = out_audio / e.audio.name
            if not link.exists():
                try:
                    link.symlink_to(e.audio)
                except (OSError, NotImplementedError):
                    # Fall back to a copy if symlinks aren't supported.
                    import shutil
                    shutil.copy2(e.audio, link)
            # Rewrite the entry's audio path to the local symlink/copy.
            e_local = AquaManifestEntry(
                audio=link, raw=e.raw, corrected=e.corrected, duration=e.duration,
                timestamp=e.timestamp, has_correction=e.has_correction,
                session_id=e.session_id, model=e.model,
            )
            train_records.append(aqua_to_nemo(e_local))
        n_val_final = len(val_entries)
        write_manifest(train_records[:len(train_entries)], train_path)
        write_manifest(train_records[len(train_entries):], val_path)
        _ = n_val_final  # kept for clarity
    else:  # messages
        train_recs = [aqua_to_messages(e) for e in train_entries]
        val_recs = [aqua_to_messages(e) for e in val_entries]
        write_messages_jsonl(train_recs, train_path)
        write_messages_jsonl(val_recs, val_path)

    summary = {
        **stats,
        "kept": len(kept),
        "train": len(train_entries),
        "val": len(val_entries),
        "audio_hours": round(audio_hours, 4),
        "target_format": target_fmt.value,
        "watermark": watermark,
    }
    (out_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
