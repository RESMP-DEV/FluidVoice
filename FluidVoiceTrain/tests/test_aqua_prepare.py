"""Tests for the Aqua/FluidVoice corpus preprocessor."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fluidvoice_finetune.config import DatasetFormat
from fluidvoice_finetune.data.aqua import (
    AquaManifestEntry, read_aqua_manifest, to_messages, to_nemo_entry,
)
from fluidvoice_finetune.data.prepare import prepare_aqua_corpus


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def aqua_corpus(tmp_path: Path) -> Path:
    """A tiny Aqua-style corpus: 4 entries (1 trailing-space-only, 1 genuine,
    1 no-correction, 1 genuine) + audio files."""
    src = tmp_path / "corpus"
    (src / "audio").mkdir(parents=True)
    entries = [
        # genuine correction (punctuation + word fix)
        {"audio": "audio/a1.wav", "raw": "its a test", "corrected": "It's a test.",
         "duration": 1.5, "timestamp": "2026-07-07T00:34:38.487Z",
         "session_id": 1, "has_correction": True},
        # trailing-space-only (should be filtered for Gemma)
        {"audio": "audio/a2.wav", "raw": "Hello.", "corrected": "Hello. ",
         "duration": 0.8, "timestamp": "2026-07-07T00:34:31.655Z",
         "session_id": 2, "has_correction": True},
        # no correction at all
        {"audio": "audio/a3.wav", "raw": "same text", "corrected": "same text",
         "duration": 1.0, "timestamp": "2026-07-07T00:34:22.870Z",
         "session_id": 3, "has_correction": False},
        # genuine correction
        {"audio": "audio/a4.wav", "raw": "cuda tile ir", "corrected": "CUDA Tile IR",
         "duration": 2.0, "timestamp": "2026-07-07T00:34:22.870Z",
         "session_id": 4, "has_correction": True},
    ]
    with (src / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    # Create the fake .wav files (empty is fine — Parakeet path will check existence).
    for e in entries:
        (src / e["audio"]).write_bytes(b"RIFF fake wav")
    return src


# --------------------------------------------------------------------------- #
# aqua.py unit tests
# --------------------------------------------------------------------------- #
def test_read_aqua_manifest_resolves_audio_paths(aqua_corpus: Path):
    entries = read_aqua_manifest(aqua_corpus / "manifest.jsonl")
    assert len(entries) == 4
    assert all(e.audio.is_absolute() for e in entries)
    assert entries[0].audio.name == "a1.wav"
    assert entries[0].raw == "its a test"
    assert entries[0].corrected == "It's a test."
    assert entries[0].duration == 1.5
    assert entries[0].session_id == 1


def test_is_genuine_correction_distinguishes_whitespace_only():
    genuine = AquaManifestEntry(audio=Path("/x.wav"), raw="hi", corrected="Hi.", duration=1.0,
                                timestamp="", has_correction=True)
    trivial = AquaManifestEntry(audio=Path("/x.wav"), raw="Hello.", corrected="Hello. ",
                                duration=1.0, timestamp="", has_correction=True)
    none = AquaManifestEntry(audio=Path("/x.wav"), raw="x", corrected="x", duration=1.0,
                             timestamp="", has_correction=False)
    assert genuine.is_genuine_correction is True
    assert trivial.is_genuine_correction is False
    assert none.is_genuine_correction is False


def test_is_user_correction_detects_manual_edit():
    with_edit = AquaManifestEntry(audio=Path("/x.wav"), raw="hi", corrected="Hi.", duration=1.0,
                                  timestamp="", has_correction=True, user_corrected="Hi there.")
    no_edit = AquaManifestEntry(audio=Path("/x.wav"), raw="hi", corrected="Hi.", duration=1.0,
                                timestamp="", has_correction=True)
    same_as_raw = AquaManifestEntry(audio=Path("/x.wav"), raw="hi", corrected="Hi.", duration=1.0,
                                    timestamp="", has_correction=True, user_corrected="hi")
    assert with_edit.is_user_correction is True
    assert no_edit.is_user_correction is False
    assert same_as_raw.is_user_correction is False  # user "edit" == raw → not a correction


def test_to_messages_builds_instruction_record():
    e = AquaManifestEntry(audio=Path("/x.wav"), raw="hi", corrected="Hi.", duration=1.0,
                          timestamp="", has_correction=True)
    rec = to_messages(e)
    assert rec["messages"][0]["role"] == "system"
    assert rec["messages"][1] == {"role": "user", "content": "hi"}
    assert rec["messages"][2] == {"role": "assistant", "content": "Hi."}


def test_to_nemo_entry_uses_corrected_text():
    e = AquaManifestEntry(audio=Path("/abs/x.wav"), raw="hi", corrected="Hi.", duration=2.5,
                          timestamp="", has_correction=True)
    entry = to_nemo_entry(e)
    assert entry["audio_filepath"] == "/abs/x.wav"
    assert entry["text"] == "Hi."
    assert entry["duration"] == 2.5


# --------------------------------------------------------------------------- #
# prepare_aqua_corpus — Gemma (messages) path
# --------------------------------------------------------------------------- #
def test_prepare_aqua_corpus_messages_filters_non_corrections(aqua_corpus: Path, tmp_path: Path):
    out = tmp_path / "out_gemma"
    summary = prepare_aqua_corpus(aqua_corpus, out, target_fmt=DatasetFormat.MESSAGES,
                                  val_split=0.0, seed=42)
    # 2 genuine corrections kept; trailing-space + no-correction filtered.
    assert summary["kept"] == 2
    assert summary["filtered_no_correction"] == 2
    assert summary["train"] == 2
    assert summary["val"] == 0
    # Records are ShareGPT messages.
    train_lines = (out / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(train_lines) == 2
    rec = json.loads(train_lines[0])
    assert "messages" in rec
    assert rec["messages"][0]["role"] == "system"


def test_prepare_aqua_corpus_writes_summary_json(aqua_corpus: Path, tmp_path: Path):
    out = tmp_path / "out"
    prepare_aqua_corpus(aqua_corpus, out, target_fmt=DatasetFormat.MESSAGES, val_split=0.0)
    summary = json.loads((out / "manifest_summary.json").read_text())
    assert summary["total"] == 4
    assert summary["kept"] == 2
    assert summary["target_format"] == "messages"


# --------------------------------------------------------------------------- #
# prepare_aqua_corpus — Parakeet (nemo-manifest) path
# --------------------------------------------------------------------------- #
def test_prepare_aqua_corpus_nemo_keeps_all_with_audio(aqua_corpus: Path, tmp_path: Path):
    out = tmp_path / "out_parakeet"
    summary = prepare_aqua_corpus(aqua_corpus, out, target_fmt=DatasetFormat.NEMO_MANIFEST,
                                  val_split=0.0, seed=42)
    # Parakeet keeps all 4 (audio exists for all); doesn't filter on correction.
    assert summary["kept"] == 4
    assert summary["filtered_no_correction"] == 0
    train_lines = (out / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(train_lines) == 4
    for line in train_lines:
        rec = json.loads(line)
        assert "audio_filepath" in rec and "text" in rec
        assert Path(rec["audio_filepath"]).is_absolute()


def test_prepare_aqua_corpus_nemo_filters_missing_audio(tmp_path: Path):
    src = tmp_path / "corpus"
    (src / "audio").mkdir(parents=True)
    # Only create audio for entry 1, not entry 2.
    (src / "audio" / "a1.wav").write_bytes(b"x")
    entries = [
        {"audio": "audio/a1.wav", "raw": "r1", "corrected": "C1.", "duration": 1.0,
         "timestamp": "2026-07-07T00:00:00Z", "has_correction": True},
        {"audio": "audio/a2.wav", "raw": "r2", "corrected": "C2.", "duration": 1.0,
         "timestamp": "2026-07-07T00:00:01Z", "has_correction": True},
    ]
    with (src / "manifest.jsonl").open("w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    out = tmp_path / "out"
    summary = prepare_aqua_corpus(src, out, target_fmt=DatasetFormat.NEMO_MANIFEST, val_split=0.0)
    assert summary["kept"] == 1
    assert summary["filtered_no_audio"] == 1


# --------------------------------------------------------------------------- #
# Watermark (incremental)
# --------------------------------------------------------------------------- #
def test_prepare_aqua_corpus_watermark_filters_old_entries(aqua_corpus: Path, tmp_path: Path):
    out = tmp_path / "out"
    # Only entries after this timestamp (none, since all are at 00:34).
    summary = prepare_aqua_corpus(aqua_corpus, out, target_fmt=DatasetFormat.MESSAGES,
                                  val_split=0.0, watermark="2026-07-08T00:00:00Z")
    assert summary["kept"] == 0
    assert summary["filtered_by_watermark"] == 4  # all 4 entries (2 genuine + 2 filtered-by-correction counted before watermark? no)
    # NOTE: watermark is applied FIRST, so all 4 are filtered by watermark before the correction filter.
    assert summary["filtered_no_correction"] == 0


def test_prepare_aqua_corpus_val_split(aqua_corpus: Path, tmp_path: Path):
    out = tmp_path / "out"
    summary = prepare_aqua_corpus(aqua_corpus, out, target_fmt=DatasetFormat.MESSAGES,
                                  val_split=0.5, seed=42)
    # 2 genuine kept → 1 train, 1 val.
    assert summary["train"] == 1
    assert summary["val"] == 1
