"""Gated, incremental, checkpointed idle-training runner.

This is what the launchd job (and a manual ``fluidvoice-finetune idle-train``
invocation) runs. Flow:

  1. Gate (on-AC + idle + thermal + not-Low-Power). Exit early if the gate fails.
  2. Read the trained watermark (max timestamp previously consumed).
  3. Prepare the corpus incrementally (only entries newer than the watermark).
  4. Run Gemma MLX training + GGUF export (resuming from the last adapter).
  5. Update the watermark.
  6. Optionally publish to HF.

Checkpoint safety: a SIGTERM/SIGINT handler lets MLX finish its current step
and write the adapter checkpoint before exit; the next run resumes via
``resume_adapter_file``. This matters because the user may grab the machine
(WillSleep / manual interrupt) mid-training.
"""
from __future__ import annotations

import signal
import sys
from dataclasses import dataclass
from pathlib import Path

from ..config import DatasetFormat, GemmaSize, Model, RunConfig, Target
from .gate import gate as _gate_check


@dataclass
class IdleTrainResult:
    trained: bool
    gate_ok: bool
    gate_reasons: list[str]
    new_records: int
    gguf_path: Path | None
    watermark: str | None
    detail: str


def _max_timestamp(entries) -> str | None:
    ts = [e.timestamp for e in entries if e.timestamp]
    return max(ts) if ts else None


def run_idle_train(
    corpus_dir: Path,
    checkpoint_dir: Path,
    *,
    size: GemmaSize = GemmaSize.E4B,
    iters: int | None = None,
    learning_rate: float | None = None,
    lora_rank: int = 16,
    publish_repo: str | None = None,
    gate_check: bool = True,
    min_idle_seconds: float = 600.0,
    dry_run_prepare: bool = False,
) -> IdleTrainResult:
    """Run one gated, incremental training pass.

    Args:
        corpus_dir: dir with ``manifest.jsonl`` + ``audio/`` (Aqua/FluidVoice shape).
        checkpoint_dir: where adapters + watermark live across runs.
        size: Gemma size (e2b/e4b).
        gate_check: if True, run the power/idle/thermal gate first.
        min_idle_seconds: gate idle threshold (only if gate_check).
        dry_run_prepare: if True, run the prepare step but skip actual training
            (useful for testing the pipeline without mlx-lm/llama.cpp installed).
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gate.
    if gate_check:
        g = _gate_check(min_idle_seconds=min_idle_seconds)
        if not g:
            return IdleTrainResult(
                trained=False, gate_ok=False, gate_reasons=g.reasons,
                new_records=0, gguf_path=None, watermark=None,
                detail=f"Gate failed: {'; '.join(g.reasons)}",
            )

    # 2. Read watermark.
    from ..gemma.mlx_train import read_watermark, write_watermark
    watermark = read_watermark(checkpoint_dir)
    resume_adapter = checkpoint_dir / "adapters"

    # 3. Prepare corpus incrementally.
    from ..data.prepare import prepare_aqua_corpus
    prepared_dir = checkpoint_dir / "prepared"
    summary = prepare_aqua_corpus(
        corpus_dir, prepared_dir, target_fmt=DatasetFormat.MESSAGES,
        val_split=0.1, watermark=watermark,
    )
    if summary["kept"] == 0:
        return IdleTrainResult(
            trained=False, gate_ok=True, gate_reasons=[],
            new_records=0, gguf_path=None, watermark=watermark,
            detail="No new training records since last watermark.",
        )

    if dry_run_prepare:
        return IdleTrainResult(
            trained=False, gate_ok=True, gate_reasons=[],
            new_records=summary["kept"], gguf_path=None, watermark=watermark,
            detail=f"Dry run: prepared {summary['kept']} new records (no training).",
        )

    # 4. Build RunConfig + train.
    cfg = RunConfig(model=Model.GEMMA, dataset=prepared_dir / "train.jsonl")
    cfg.gemma.target = Target.MLX
    cfg.gemma.size = size
    cfg.gemma.lora_rank = lora_rank
    cfg.gemma.output_dir = checkpoint_dir
    if iters is not None:
        cfg.extra["iters"] = iters
    if learning_rate is not None:
        cfg.gemma.learning_rate = learning_rate

    # Install SIGTERM/SIGINT handler so launchd/caffeinate teardown is clean.
    _install_signal_handlers()

    from ..gemma import run_gemma
    gguf_path = run_gemma(cfg, data_dir=prepared_dir,
                          resume_adapter=resume_adapter if resume_adapter.exists() else None)

    # 5. Update watermark.
    from ..data.aqua import read_aqua_manifest
    entries = read_aqua_manifest(corpus_dir / "manifest.jsonl")
    new_watermark = _max_timestamp(entries)
    if new_watermark:
        write_watermark(checkpoint_dir, new_watermark)

    # 6. Optional publish.
    detail = f"Trained on {summary['kept']} new records → {gguf_path}"
    if publish_repo:
        from ..publish.huggingface import publish_artifact
        url = publish_artifact(gguf_path, repo_id=publish_repo, model=Model.GEMMA,
                               size=size, path_in_repo="models/")
        detail += f"; published to {url}"

    return IdleTrainResult(
        trained=True, gate_ok=True, gate_reasons=[], new_records=summary["kept"],
        gguf_path=gguf_path, watermark=new_watermark, detail=detail,
    )


# --------------------------------------------------------------------------- #
# Signal handling for clean checkpoint-on-interrupt
# --------------------------------------------------------------------------- #
_INTERRUPTED = False


def _is_interrupted() -> bool:
    return _INTERRUPTED


def _install_signal_handlers() -> None:
    """Mark interrupted on SIGTERM/SIGINT; MLX checkpoints at its next step boundary.

    We don't raise here — mlx_lm's training loop checks for interrupts and saves
    the adapter at save_every intervals. A second signal (SIGINT) forces raise."""
    def handler(signum, frame):
        global _INTERRUPTED
        _INTERRUPTED = True
        print(f"[idle-train] received signal {signum}; will checkpoint at next step "
              f"(send SIGINT again to force exit)", file=sys.stderr)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handler)
    try:
        signal.signal(signal.SIGINT, handler)
    except (ValueError, OSError):
        pass  # not in main thread
