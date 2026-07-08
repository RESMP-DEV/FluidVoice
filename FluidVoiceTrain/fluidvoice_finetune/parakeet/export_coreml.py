"""Export a fine-tuned NeMo Parakeet checkpoint to a CoreML ``.mlmodelc`` bundle.

NeMo → ONNX → CoreML, matching the layout the FluidAudio package expects under
the ``FluidInference/*-coreml`` HF org (compute units: cpuAndNeuralEngine).
"""
from __future__ import annotations

from pathlib import Path

from ..config import RunConfig


_COREML_COMPUTE_MAP = {
    "cpuAndNeuralEngine": "all",          # coremltools ComputeUnit.ALL
    "cpuAndGPU": "cpuAndGPU",
    "cpuOnly": "cpuOnly",
}


def export_coreml(nemo_ckpt: Path, cfg: RunConfig) -> Path:
    """Export ``nemo_ckpt`` → ``<output_dir>/parakeet-finetuned.mlmodelc``."""
    try:
        import coremltools as ct  # noqa: F401
        import onnx  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "CoreML export requires the [parakeet] extra: uv pip install -e .[parakeet]"
        ) from e

    p = cfg.parakeet
    out_dir = p.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    mlmodelc_path = out_dir / "parakeet-finetuned.mlmodelc"

    # NOTE: sketched sequence — pin nemo_toolkit / coremltools versions and adapt.
    #
    # Step 1: NeMo → ONNX.
    #   from nemo.collections.asr.models import ASRModel
    #   model = ASRModel.restore_from(str(nemo_ckpt))
    #   model.export(str(onnx_dir / "parakeet.onnx"))   # NeMo's built-in ONNX export
    #
    # Step 2: ONNX → CoreML via coremltools.
    #   import coremltools as ct
    #   mlmodel = ct.converters.onnx.convert(
    #       model=str(onnx_dir / "parakeet.onnx"),
    #       compute_units=_COREML_COMPUTE_MAP[p.coreml_compute_units],
    #       minimum_deployment_target=ct.target.macOS15,
    #   )
    #   mlmodel.save(str(mlmodelc_path))

    _write_export_manifest(out_dir, cfg, mlmodelc_path)
    return mlmodelc_path


def _write_export_manifest(out_dir: Path, cfg: RunConfig, mlmodelc_path: Path) -> None:
    import json
    manifest = {
        "format": "coreml",
        "compute_units": cfg.parakeet.coreml_compute_units,
        "output": str(mlmodelc_path),
        "note": "scaffold: populate with real NeMo ONNX export + coremltools convert before a live run",
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
