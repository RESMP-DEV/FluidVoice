"""Tests for the data formatters and prepare pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fluidvoice_finetune.config import DatasetFormat
from fluidvoice_finetune.data.gemma_format import (
    read_messages_jsonl,
    to_messages,
    write_messages_jsonl,
)
from fluidvoice_finetune.data.parakeet_format import (
    read_manifest,
    to_manifest_entry,
    total_duration_hours,
    write_manifest,
)
from fluidvoice_finetune.data.prepare import prepare_dataset


# --------------------------------------------------------------------------- #
# gemma_format
# --------------------------------------------------------------------------- #
def test_to_messages_builds_sharegpt_record():
    rec = to_messages("hello world", "Hello, world.")
    assert rec == {
        "messages": [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "Hello, world."},
        ]
    }


def test_to_messages_rejects_empty():
    with pytest.raises(ValueError):
        to_messages("", "target")
    with pytest.raises(ValueError):
        to_messages("raw", "")


def test_write_and_read_messages_jsonl_roundtrip(tmp_path: Path):
    recs = [to_messages("a", "A"), to_messages("b", "B")]
    out = tmp_path / "x.jsonl"
    n = write_messages_jsonl(recs, out)
    assert n == 2
    got = read_messages_jsonl(out)
    assert got == recs


# --------------------------------------------------------------------------- #
# parakeet_format
# --------------------------------------------------------------------------- #
def test_to_manifest_entry_requires_text():
    with pytest.raises(ValueError):
        to_manifest_entry("/x.wav", "")


def test_to_manifest_entry_without_duration_when_no_soundfile(tmp_path: Path, monkeypatch):
    # Force soundfile to be absent so duration is None.
    import sys
    monkeypatch.setitem(sys.modules, "soundfile", None)
    entry = to_manifest_entry(tmp_path / "nonexistent.wav", "hi", duration=None)
    assert entry["audio_filepath"].endswith("nonexistent.wav")
    assert entry["text"] == "hi"
    assert "duration" not in entry  # couldn't compute, file missing


def test_to_manifest_entry_with_explicit_duration():
    entry = to_manifest_entry("/data/a.wav", "hi", duration=2.5)
    assert entry["duration"] == 2.5


def test_write_and_read_manifest_roundtrip(tmp_path: Path):
    entries = [
        to_manifest_entry("/a.wav", "hi", duration=1.0),
        to_manifest_entry("/b.wav", "yo", duration=2.0),
    ]
    out = tmp_path / "m.jsonl"
    n = write_manifest(entries, out)
    assert n == 2
    got = read_manifest(out)
    assert got == entries
    assert total_duration_hours(got) == pytest.approx(3.0 / 3600.0)


def test_write_manifest_rejects_missing_keys(tmp_path: Path):
    with pytest.raises(ValueError):
        write_manifest([{"audio_filepath": "/a.wav"}], tmp_path / "bad.jsonl")


# --------------------------------------------------------------------------- #
# prepare_dataset (end-to-end ingest → split)
# --------------------------------------------------------------------------- #
def test_prepare_dataset_raw_csv_to_messages(raw_pairs_csv: Path, tmp_path: Path):
    out_dir = tmp_path / "out"
    count = prepare_dataset(
        raw_pairs_csv, out_dir, target_fmt=DatasetFormat.MESSAGES,
        source_fmt=DatasetFormat.RAW, val_split=0.0, seed=0,
    )
    assert count == 3  # 3 data rows (header skipped by DictReader fieldnames)
    train = (out_dir / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert all(json.loads(r)["messages"][0]["role"] == "user" for r in train)


def test_prepare_dataset_messages_passthrough(messages_jsonl: Path, tmp_path: Path):
    out_dir = tmp_path / "out"
    count = prepare_dataset(
        messages_jsonl, out_dir, target_fmt=DatasetFormat.MESSAGES,
        source_fmt=DatasetFormat.AUTO, val_split=0.5, seed=0,
    )
    assert count == 2
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "valid.jsonl").exists()


def test_prepare_dataset_nemo_manifest_passthrough(nemo_manifest_jsonl: Path, tmp_path: Path):
    out_dir = tmp_path / "out"
    count = prepare_dataset(
        nemo_manifest_jsonl, out_dir, target_fmt=DatasetFormat.NEMO_MANIFEST,
        source_fmt=DatasetFormat.AUTO, val_split=0.0, seed=0,
    )
    assert count == 2
    train = (out_dir / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert all("audio_filepath" in json.loads(r) for r in train)


def test_prepare_dataset_splits_respect_val_split(raw_pairs_csv: Path, tmp_path: Path):
    out_dir = tmp_path / "out"
    count = prepare_dataset(
        raw_pairs_csv, out_dir, target_fmt=DatasetFormat.MESSAGES,
        source_fmt=DatasetFormat.RAW, val_split=0.34, seed=42,  # ~1 of 3 in val
    )
    train_n = len((out_dir / "train.jsonl").read_text().strip().splitlines())
    val_n = len((out_dir / "valid.jsonl").read_text().strip().splitlines())
    assert train_n + val_n == count
    assert val_n == 1
