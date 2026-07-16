from __future__ import annotations

import os
from numbers import Integral

import torch

from .protocol import PROTOCOL_VERSION


def _validated_indices(indices, *, minimum_count: int, label: str) -> list[int]:
    if isinstance(indices, torch.Tensor):
        if indices.is_floating_point() and not torch.equal(indices, indices.round()):
            raise ValueError(f"{label} contains non-integral values.")
        indices = indices.flatten().tolist()
    if not isinstance(indices, list):
        raise TypeError(f"{label} must be stored as a list or tensor.")
    if any(
        isinstance(index, bool)
        or not isinstance(index, Integral)
        or int(index) < 0
        for index in indices
    ):
        raise ValueError(f"{label} must contain nonnegative integers.")
    parsed = [int(index) for index in indices]
    if len(parsed) < minimum_count or len(set(parsed)) != len(parsed):
        raise ValueError(
            f"{label} needs at least {minimum_count} unique source indices."
        )
    return parsed


def load_reference_bank_bundle(path: str) -> tuple[torch.Tensor, dict]:
    """Load a preprocessed reference bank in model input coordinates.

    The file may contain a tensor directly or a dictionary with an ``images``
    tensor. Banks are kept on CPU and sampled lazily to avoid occupying GPU
    memory for references that are not used in the current minibatch.
    """
    if not path:
        raise ValueError("A non-empty reference-bank path is required.")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    images = payload.get("images") if isinstance(payload, dict) else payload
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if not isinstance(images, torch.Tensor):
        raise TypeError("Reference bank must be a tensor or contain tensor key 'images'.")
    if images.dim() != 4 or images.size(0) < 2:
        raise ValueError("Reference bank must have shape [N,C,H,W] with N >= 2.")
    if not torch.isfinite(images).all():
        raise ValueError("Reference bank contains NaN or infinite values.")
    if not isinstance(metadata, dict):
        raise TypeError("Reference-bank metadata must be a dictionary.")
    return images.detach().contiguous().cpu(), metadata


def load_reference_bank(path: str) -> torch.Tensor:
    images, _ = load_reference_bank_bundle(path)
    return images


def load_index_reservation(path: str) -> tuple[list[int], dict]:
    if not path:
        raise ValueError("A non-empty reservation path is required.")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    indices = payload.get("indices") if isinstance(payload, dict) else payload
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    parsed = _validated_indices(
        indices, minimum_count=4, label="Reservation"
    )
    if not isinstance(metadata, dict):
        raise TypeError("Reservation metadata must be a dictionary.")
    required = {"dataset", "input_profile", "source_split", "selection", "seed"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Reservation metadata is missing: {missing}")
    if str(metadata["source_split"]).lower() != "train":
        raise ValueError("Reference candidates must be reserved from the train split.")
    if metadata["selection"] != "model_independent_random_reservation":
        raise ValueError(
            f"Protocol-v{PROTOCOL_VERSION} requires a model-independent random "
            "candidate reservation."
        )
    if str(metadata["input_profile"]).lower() not in {"native", "imagenet"}:
        raise ValueError("Reservation input_profile must be native or imagenet.")
    if (
        isinstance(metadata["seed"], bool)
        or not isinstance(metadata["seed"], Integral)
    ):
        raise ValueError("Reservation seed must be an integer.")
    return parsed, metadata


def reference_source_indices(
    metadata: dict,
    expected_count: int,
    expected_role: str,
) -> list[int]:
    required = {"dataset", "input_profile", "source_split", "role", "source_indices"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Reference-bank metadata is missing: {missing}")
    if metadata["source_split"] != "train":
        raise ValueError("Protocol-v2 reference banks must come from the train split.")
    if metadata["role"] != expected_role:
        raise ValueError(
            f"Expected a {expected_role} reference bank, found {metadata['role']}."
        )
    indices = _validated_indices(
        metadata["source_indices"],
        minimum_count=2,
        label="Reference bank",
    )
    if len(indices) != int(expected_count):
        raise ValueError(
            "Reference-bank source index count does not match the image tensor."
        )
    return indices


def sample_reference_baselines(
    x: torch.Tensor,
    bank: torch.Tensor,
    count: int,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Sample distinct references for each batch element without replacement."""
    if bank.dim() != 4 or tuple(bank.shape[1:]) != tuple(x.shape[1:]):
        raise ValueError(
            "Reference-bank image shape does not match model input: "
            f"bank={tuple(bank.shape[1:])}, input={tuple(x.shape[1:])}."
        )
    if bank.size(0) < 2:
        raise ValueError("Reference bank must contain at least two images.")
    count = min(max(int(count), 2), bank.size(0))
    index_device = generator.device if generator is not None else torch.device("cpu")
    indices = torch.stack(
        [
            torch.randperm(
                bank.size(0), device=index_device, generator=generator
            )[:count]
            for _ in range(x.size(0))
        ],
        dim=0,
    ).cpu()
    sampled = {}
    for index in range(count):
        sampled[f"bank_{index}"] = bank[indices[:, index]].to(
            device=x.device, dtype=x.dtype, non_blocking=True
        )
    return sampled
