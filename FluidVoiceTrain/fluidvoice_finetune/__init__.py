"""FluidVoice fine-tuning pipeline.

Unified SFT/LoRA fine-tuning for the FluidVoice stack:

- **Gemma** (the "Fluid Intelligence" on-device enhancement LLM, a modified
  **Gemma 4** derivative — confirmed from the GGUF metadata
  (``general.architecture = "gemma4"``) — shipped as `fluid-1-*-q4_k_m.gguf`
  at ``altic-dev/FluidIntelligence``). Supports two sizes via the ``--size``
  flag: ``e2b`` (``google/gemma-4-E2B``) and ``e4b`` (``google/gemma-4-E4B``).
- **Parakeet** (on-device ASR; NVIDIA NeMo CTC/TDT checkpoints re-exported to
  CoreML ``.mlmodelc`` bundles for the ``FluidInference/*-coreml`` repos).

The CLI dispatches by ``--model {gemma,parakeet}``; Gemma further branches on
``--target {mlx,cuda}`` (local Apple Silicon vs. remote GPU). Parakeet is
CUDA-only.
"""
from __future__ import annotations

__version__ = "0.1.0"
