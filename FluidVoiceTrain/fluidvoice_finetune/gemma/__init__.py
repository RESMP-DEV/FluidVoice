"""Gemma (Fluid Intelligence) fine-tuning entrypoint.

Dispatches to :mod:`mlx_train` (Apple Silicon) or :mod:`transformers_train`
(CUDA) based on ``RunConfig.gemma.target``, then runs :mod:`export_gguf` to
produce the ``fluid-1-<size>-q4_k_m.gguf`` artifact the FluidVoice Swift app
expects.

The MLX path (the on-device / idle-train path) is: ``mlx_lm.lora`` →
``mlx_lm.fuse`` → ``convert_hf_to_gguf.py`` → ``llama-quantize``.
"""
from __future__ import annotations

from pathlib import Path

from ..config import RunConfig, Target


def run_gemma(cfg: RunConfig, data_dir: Path | None = None,
              resume_adapter: Path | None = None) -> Path:
    """Run Gemma fine-tuning + GGUF export. Returns the path to the GGUF file."""
    if cfg.gemma.target is Target.MLX:
        from .mlx_train import train as mlx_train
        merged_dir = mlx_train(cfg, data_dir=data_dir, resume_adapter=resume_adapter)
    else:
        from .transformers_train import train as transformers_train
        merged_dir = transformers_train(cfg, data_dir=data_dir)

    from .export_gguf import export_gguf
    gguf_path = export_gguf(merged_dir, cfg)
    return gguf_path
