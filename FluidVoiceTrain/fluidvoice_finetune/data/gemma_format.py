"""Gemma message-format converter.

Targets the ShareGPT / OpenAI ``messages`` convention used by DistillKit and
``fluid-1`` (the Fluid Intelligence Gemma derivative). Output is JSONL with one
record per line:

    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

For Fluid Intelligence specifically the task is *dictation enhancement*:
``user`` = the raw ASR transcript (possibly messy), ``assistant`` = the
cleaned/formatted target text. The exact instruction phrasing is applied at
training time via the chat template; this module only shapes the data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _record_from_pair(raw: str, target: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": raw.strip()},
            {"role": "assistant", "content": target.strip()},
        ]
    }


def to_messages(raw_text: str, target_text: str) -> dict[str, Any]:
    """Build a single ShareGPT-style record from a raw/target pair."""
    if not raw_text.strip() or not target_text.strip():
        raise ValueError("raw_text and target_text must both be non-empty")
    return _record_from_pair(raw_text, target_text)


def write_messages_jsonl(records: Iterable[dict[str, Any]], out_path: Path) -> int:
    """Write records as JSONL. Returns the number of records written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")
            count += 1
    return count


def read_messages_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a messages-JSONL file back into a list of records."""
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
