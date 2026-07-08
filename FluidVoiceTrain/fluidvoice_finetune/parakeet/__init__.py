"""Parakeet ASR fine-tuning entrypoint (NeMo on CUDA, then CoreML export)."""
from __future__ import annotations

from pathlib import Path

from ..config import RunConfig


def run_parakeet(cfg: RunConfig) -> Path:
    """Run NeMo fine-tuning + CoreML export. Returns the .mlmodelc bundle path."""
    from .nemo_train import train as nemo_train
    nemo_ckpt = nemo_train(cfg)

    from .export_coreml import export_coreml
    mlmodelc_path = export_coreml(nemo_ckpt, cfg)
    return mlmodelc_path
