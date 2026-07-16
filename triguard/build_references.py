"""Build disjoint, calibration-relative reference banks from a reserved train split."""

import argparse
import hashlib
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import get_dataset
from .models import get_model, uses_imagenet_preprocessing
from .references import load_index_reservation


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _collect_images(dataset, indices):
    return torch.stack([dataset[int(index)][0] for index in indices], dim=0)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_calibration_checkpoint(
    checkpoint: str,
    candidate_indices: str,
    dataset: str,
    model: str,
):
    sidecar = f"{checkpoint}.meta.json"
    if not os.path.exists(sidecar):
        raise ValueError(
            f"Missing calibration metadata sidecar: {sidecar}. "
            "Recreate the checkpoint with run_triguard.py."
        )
    with open(sidecar, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("Calibration checkpoint hash does not match its metadata.")
    if str(metadata.get("dataset", "")).lower() != dataset.lower():
        raise ValueError("Calibration checkpoint was trained for a different dataset.")
    if str(metadata.get("model", "")).lower() != model.lower():
        raise ValueError("Calibration checkpoint was trained for a different model.")
    expected_reservation = _sha256(candidate_indices)
    if metadata.get("training_reservation_sha256") != expected_reservation:
        raise ValueError(
            "Calibration checkpoint was not trained with the supplied candidate "
            "reservation excluded."
        )
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Frozen calibration checkpoint used to score target-task neutrality.",
    )
    parser.add_argument("--out", default="reference_banks")
    parser.add_argument("--bank_size", type=int, default=64)
    parser.add_argument("--heldout_size", type=int, default=64)
    parser.add_argument("--candidate_limit", type=int, default=2048)
    parser.add_argument(
        "--selection_balance", choices=["label", "none"], default="label"
    )
    parser.add_argument(
        "--candidate_indices",
        default="",
        help="Model-independent reservation excluded from calibration training.",
    )
    parser.add_argument("--allow_unreserved_candidates", action="store_true")
    parser.add_argument("--allow_unverified_checkpoint", action="store_true")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    if args.bank_size < 2 or args.heldout_size < 2:
        parser.error("--bank_size and --heldout_size must each be at least 2.")
    if not args.candidate_indices and not args.allow_unreserved_candidates:
        parser.error(
            "--candidate_indices is required for leakage-free bank construction. "
            "Use --allow_unreserved_candidates only for exploratory work."
        )
    if args.batch <= 0 or args.num_workers < 0:
        parser.error("--batch must be positive and --num_workers cannot be negative.")

    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_profile = (
        "imagenet" if uses_imagenet_preprocessing(args.model) else "native"
    )
    train_set, test_set, _, _, _, meta = get_dataset(
        args.dataset,
        data_root=args.data_root,
        input_profile=input_profile,
    )
    # Score and materialize reserved training examples with the deterministic
    # evaluation transform, never the stochastic training augmentation.
    train_set.transform = test_set.transform

    needed = int(args.bank_size) + int(args.heldout_size)
    if needed > len(train_set):
        parser.error(
            "The requested train and held-out banks exceed the training-set size."
        )
    if args.candidate_indices:
        reserved, reservation_metadata = load_index_reservation(
            args.candidate_indices
        )
        if reservation_metadata.get("dataset", args.dataset.lower()) != args.dataset.lower():
            parser.error("Candidate reservation was built for a different dataset.")
        if reservation_metadata.get("input_profile", input_profile) != input_profile:
            parser.error("Candidate reservation uses a different input profile.")
        if max(reserved) >= len(train_set):
            parser.error("Candidate reservation contains an out-of-range index.")
        candidates = torch.tensor(reserved, dtype=torch.long)
        candidate_count = len(reserved)
    else:
        candidate_count = min(max(int(args.candidate_limit), needed), len(train_set))
        generator = torch.Generator().manual_seed(args.seed)
        candidates = torch.randperm(len(train_set), generator=generator)[:candidate_count]
    if needed > candidate_count:
        parser.error("The reserved candidate pool is smaller than the requested banks.")
    if args.candidate_indices and not args.allow_unverified_checkpoint:
        try:
            verify_calibration_checkpoint(
                args.checkpoint,
                args.candidate_indices,
                args.dataset,
                args.model,
            )
        except ValueError as exc:
            parser.error(str(exc))

    model = get_model(
        args.model, args.dataset, num_classes=int(meta["num_classes"])
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(
        Subset(train_set, candidates.tolist()),
        batch_size=args.batch,
        shuffle=False,
        num_workers=max(int(args.num_workers), 0),
    )
    neutrality = []
    candidate_labels = []
    with torch.no_grad():
        for images, labels in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1)
            # KL(p || Uniform) = log(C) - H(p); lower is more neutral relative
            # to the frozen calibration model.
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)
            neutrality.append(
                (np.log(probabilities.size(1)) - entropy).detach().cpu()
            )
            candidate_labels.append(labels.detach().cpu().long())
    scores = torch.cat(neutrality)
    labels = torch.cat(candidate_labels)
    global_order = torch.argsort(scores)
    if args.selection_balance == "label":
        per_class = {
            int(label): positions[torch.argsort(scores[positions])].tolist()
            for label in labels.unique(sorted=True).tolist()
            for positions in [(labels == int(label)).nonzero(as_tuple=False).flatten()]
        }
        selected_positions = []
        rank = 0
        while len(selected_positions) < needed:
            added = False
            for label in sorted(per_class):
                if rank < len(per_class[label]):
                    selected_positions.append(per_class[label][rank])
                    added = True
                    if len(selected_positions) == needed:
                        break
            if not added:
                break
            rank += 1
        if len(selected_positions) < needed:
            used = set(selected_positions)
            selected_positions.extend(
                int(position)
                for position in global_order.tolist()
                if int(position) not in used
            )
        order = torch.tensor(selected_positions[:needed], dtype=torch.long)
    else:
        order = global_order[:needed]
    selected = candidates[order]
    selected_scores = scores[order]
    selected_labels = labels[order]
    split_generator = torch.Generator().manual_seed(args.seed + 1)
    split_order = torch.randperm(needed, generator=split_generator)
    bank_positions = split_order[: args.bank_size]
    heldout_positions = split_order[
        args.bank_size : args.bank_size + args.heldout_size
    ]
    bank_indices = selected[bank_positions].tolist()
    heldout_indices = selected[heldout_positions].tolist()

    os.makedirs(args.out, exist_ok=True)
    common = {
        "dataset": args.dataset.lower(),
        "input_profile": input_profile,
        "source_split": "train",
        "selection": (
            "label_balanced_lowest_kl_to_uniform"
            if args.selection_balance == "label"
            else "lowest_kl_to_uniform"
        ),
        "selection_model": args.model.lower(),
        "selection_checkpoint": os.path.abspath(args.checkpoint),
        "selection_checkpoint_sha256": _sha256(args.checkpoint),
        "candidate_count": candidate_count,
        "candidate_reservation_sha256": (
            _sha256(args.candidate_indices) if args.candidate_indices else ""
        ),
        "seed": args.seed,
    }
    paths = []
    role_specs = [
        (
            "train",
            bank_indices,
            selected_scores[bank_positions].tolist(),
            selected_labels[bank_positions].tolist(),
        ),
        (
            "heldout",
            heldout_indices,
            selected_scores[heldout_positions].tolist(),
            selected_labels[heldout_positions].tolist(),
        ),
    ]
    for role, indices, role_scores, role_labels in role_specs:
        path = os.path.join(
            args.out,
            f"{args.dataset.lower()}_{args.model.lower()}_{role}_references.pt",
        )
        temporary = f"{path}.tmp"
        torch.save(
            {
                "images": _collect_images(train_set, indices),
                "metadata": {
                    **common,
                    "role": role,
                    "source_indices": indices,
                    "source_labels": role_labels,
                    "neutrality_kl": role_scores,
                },
            },
            temporary,
        )
        os.replace(temporary, path)
        paths.append(path)
    print("Wrote reference banks:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
