"""Shared pytest fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def raw_pairs_csv(tmp_path: Path) -> Path:
    """A tiny raw,target CSV in the Aqua-dictionary style (but as text pairs).
    Targets containing commas are quoted per CSV rules."""
    p = tmp_path / "pairs.csv"
    p.write_text(
        "raw,target\n"
        'hello world how are you,"Hello world, how are you?"\n'
        "its a test its only a test,It's a test. It's only a test.\n"
        "cuda tile ir,CUDA Tile IR\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def messages_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "messages.jsonl"
    records = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hi!"}]},
        {"messages": [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "Bye!"}]},
    ]
    with p.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return p


@pytest.fixture
def nemo_manifest_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "manifest.jsonl"
    records = [
        {"audio_filepath": "/data/a1.wav", "text": "hello", "duration": 1.2},
        {"audio_filepath": "/data/a2.wav", "text": "world", "duration": 0.8},
    ]
    with p.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return p
