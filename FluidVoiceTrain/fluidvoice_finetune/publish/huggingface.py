"""HuggingFace Hub upload for FluidVoice artifacts.

Distinguished from the tracesmith (agent-trace-share) uploader: that one pushes
*datasets* (``repo_type='dataset'``); this one pushes *models*
(``repo_type='model'``) — GGUF for Gemma, ``.mlmodelc`` for Parakeet.
"""
from __future__ import annotations

from pathlib import Path

from ..config import GemmaSize, Model
from .manifest import render_card


def publish_artifact(
    artifact: Path,
    repo_id: str,
    model: Model,
    size: GemmaSize | None = None,
    base_checkpoint: str | None = None,
    path_in_repo: str | None = None,
    private: bool = False,
    token: str | None = None,
    dataset_provenance: str | None = None,
    training_notes: str | None = None,
) -> str:
    """Upload an artifact (GGUF / .mlmodelc / .nemo) to HF Hub with a card.

    Returns the Hub URL. Requires the ``huggingface_hub`` dependency (in the
    core deps of this package).
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as e:
        raise RuntimeError("install with: pip install -e . (huggingface_hub is a core dep)") from e

    if not artifact.exists():
        raise FileNotFoundError(artifact)

    api = HfApi(token=token)
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True, token=token)

    # Render and upload the model card.
    card = render_card(
        artifact, model, repo_id, size=size, base_checkpoint=base_checkpoint,
        dataset_provenance=dataset_provenance, training_notes=training_notes,
    )
    tmp_card = artifact.parent / "README.md" if artifact.is_file() else artifact / "README.md"
    tmp_card.write_text(card)
    api.upload_file(
        path_or_fileobj=str(tmp_card),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )

    # Upload the artifact itself.
    if artifact.is_file():
        target = f"{path_in_repo or ''}{artifact.name}".lstrip("/")
        api.upload_file(
            path_or_fileobj=str(artifact),
            path_in_repo=target,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
    else:
        # Directory (e.g. a .mlmodelc bundle). path_in_repo becomes the subdir.
        api.upload_folder(
            folder_path=str(artifact),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=path_in_repo or ".",
            token=token,
        )

    return f"https://huggingface.co/{repo_id}"
