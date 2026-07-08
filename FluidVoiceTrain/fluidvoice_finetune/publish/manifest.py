"""Model-card generation for FluidVoice artifacts.

Mirrors the existing ``altic-dev/FluidIntelligence`` card conventions:
- Gemma derivatives carry the "modified Gemma model derivative — Gemma Terms of
  Use apply" license note.
- Parakeet CoreML exports are NVIDIA NeMo derivatives under NeMo's license.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..config import GemmaSize, Model


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_size_bytes(path: Path) -> int:
    return path.stat().st_size


def render_gemma_card(
    gguf_path: Path,
    size: GemmaSize,
    repo_id: str,
    dataset_provenance: str | None = None,
    training_notes: str | None = None,
) -> str:
    sha = sha256_file(gguf_path)
    n_bytes = file_size_bytes(gguf_path)
    filename = gguf_path.name
    lines = [
        f"# {repo_id}",
        "",
        "FluidIntelligence model artifacts for local on-device use.",
        "",
        "## Model",
        f"- `models/{filename}`",
        f"- Base: `{size.base_checkpoint}` (fine-tuned derivative)",
        "- Quantization: Q4_K_M (GGUF)",
        f"- SHA-256: `{sha}`",
        f"- Size: `{n_bytes}` bytes (~{n_bytes / (1024**3):.2f} GB)",
    ]
    if dataset_provenance:
        lines += ["", "## Training data", dataset_provenance]
    if training_notes:
        lines += ["", "## Training", training_notes]
    lines += [
        "",
        "## License",
        "This model is a modified Gemma model derivative. Gemma Terms of Use apply.",
        "",
        f"Loaded by FluidVoice as the `{size.value.upper()}` Fluid Intelligence variant "
        f"(`artifactFilename = {size.artifact_filename}`).",
    ]
    return "\n".join(lines) + "\n"


def render_parakeet_card(
    bundle_path: Path,
    repo_id: str,
    base_checkpoint: str,
    dataset_provenance: str | None = None,
    training_notes: str | None = None,
) -> str:
    lines = [
        f"# {repo_id}",
        "",
        "Fine-tuned Parakeet ASR model, exported to CoreML for FluidVoice on-device use.",
        "",
        "## Model",
        f"- Base: `{base_checkpoint}` (NeMo fine-tuned derivative)",
        "- Format: CoreML `.mlmodelc` bundle (compute units: cpuAndNeuralEngine)",
    ]
    if dataset_provenance:
        lines += ["", "## Training data", dataset_provenance]
    if training_notes:
        lines += ["", "## Training", training_notes]
    lines += [
        "",
        "## License",
        "NVIDIA NeMo / Parakeet license terms apply to this derivative.",
    ]
    return "\n".join(lines) + "\n"


def render_card(
    artifact_path: Path,
    model: Model,
    repo_id: str,
    size: GemmaSize | None = None,
    base_checkpoint: str | None = None,
    **extra: Any,
) -> str:
    if model is Model.GEMMA:
        if size is None:
            raise ValueError("size is required for Gemma cards")
        return render_gemma_card(artifact_path, size, repo_id, **extra)
    return render_parakeet_card(artifact_path, repo_id, base_checkpoint or "", **extra)
