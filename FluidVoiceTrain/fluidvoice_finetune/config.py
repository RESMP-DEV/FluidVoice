"""Run-time configuration dataclasses.

These describe a single fine-tuning run. The CLI builds a :class:`RunConfig`
from its flags and passes it down to the chosen backend. Keeping the config in
plain dataclasses (rather than scattered kwargs) makes the train scripts
testable without importing heavy ML deps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Model(str, Enum):
    GEMMA = "gemma"
    PARAKEET = "parakeet"


class Target(str, Enum):
    MLX = "mlx"       # Apple Silicon, local
    CUDA = "cuda"     # NVIDIA GPU, remote


class GemmaSize(str, Enum):
    """Gemma-3n base-checkpoint size. Selects the HF source and the output
    GGUF filename consumed by the FluidVoice Swift toggle:

    ``fluid-1-<size>-q4_k_m.gguf`` (e.g. ``fluid-1-e4b-q4_k_m.gguf``).
    """

    E2B = "e2b"
    E4B = "e4b"

    @property
    def base_checkpoint(self) -> str:
        return {
            GemmaSize.E2B: "google/gemma-3n-e2b",
            GemmaSize.E4B: "google/gemma-3n-e4b",
        }[self]

    @property
    def artifact_filename(self) -> str:
        """GGUF filename the FluidVoice Swift app expects for this variant.

        Mirrors ``FluidIntelligenceModelVariant.artifactFilename`` in
        ``Sources/Fluid/Services/PrivateAIProvider.swift``.
        """
        return f"fluid-1-{self.value}-q4_k_m.gguf"


class DatasetFormat(str, Enum):
    AUTO = "auto"
    MESSAGES = "messages"        # ShareGPT / OpenAI chat messages
    NEMO_MANIFEST = "nemo-manifest"  # {audio_filepath, text, duration}
    RAW = "raw"                  # opaque; backend decides


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #
@dataclass
class GemmaConfig:
    size: GemmaSize = GemmaSize.E4B
    target: Target = Target.MLX
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 3
    per_device_batch_size: int = 1
    grad_accum_steps: int = 8
    max_seq_length: int = 2048
    quantize_base: bool = True  # 4-bit base (CUDA/bnb only); MLX ignores this
    # Output
    output_dir: Path = field(default_factory=lambda: Path("./output/gemma"))
    quant: str = "q4_k_m"       # GGUF quantization

    @property
    def base_checkpoint(self) -> str:
        return self.size.base_checkpoint


@dataclass
class ParakeetConfig:
    """Parakeet ASR fine-tuning config.

    The base checkpoint is a NeMo-ready Parakeet model. Fine-tuning produces a
    new ``.nemo`` checkpoint, which is then exported to ONNX → CoreML ``.mlmodelc``
    matching the layout the FluidAudio package expects under the
    ``FluidInference/*-coreml`` HF org.
    """

    base_checkpoint: str = "nvidia/parakeet-tdt-0.6b-v2"
    learning_rate: float = 1e-4
    num_epochs: int = 5
    per_device_batch_size: int = 16
    grad_accum_steps: int = 1
    # Export
    output_dir: Path = field(default_factory=lambda: Path("./output/parakeet"))
    coreml_compute_units: str = "cpuAndNeuralEngine"  # matches FluidAudio usage


@dataclass
class PublishConfig:
    repo_id: str | None = None        # e.g. "altic-dev/FluidIntelligence" or "FluidInference/parakeet-...-finetuned-coreml"
    path_in_repo: str | None = None   # e.g. "models/" for Gemma GGUF
    upload_token: str | None = None
    private: bool = False
    skip_upload: bool = False


@dataclass
class RunConfig:
    model: Model
    dataset: Path
    dataset_format: DatasetFormat = DatasetFormat.AUTO
    val_split: float = 0.05
    seed: int = 42
    gemma: GemmaConfig = field(default_factory=GemmaConfig)
    parakeet: ParakeetConfig = field(default_factory=ParakeetConfig)
    publish: PublishConfig = field(default_factory=PublishConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Cross-field validation. Raises ValueError on incompatible combos."""
        if self.model is Model.PARAKEET and self.parakeet is not None:
            # Parakeet is CUDA-only; reject an explicit MLX target if someone sets it
            # via extra. (ParakeetConfig has no `target` field by design.)
            target = self.extra.get("target")
            if target == Target.MLX.value:
                raise ValueError(
                    "Parakeet fine-tuning requires NeMo on a CUDA host; "
                    "--target mlx is not supported for --model parakeet."
                )
        if self.val_split < 0 or self.val_split >= 1:
            raise ValueError(f"val_split must be in [0, 1), got {self.val_split}")
