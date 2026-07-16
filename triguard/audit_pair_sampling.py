import argparse
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .attributions import attribution_allocation_distance, integrated_gradients
from .data import get_dataset
from .eval import _baseline_family, _torch_generator
from .models import get_model, uses_imagenet_preprocessing
from .protocol import PROTOCOL_VERSION
from .references import load_reference_bank_bundle


def aggregate_pair_risk(
    pair_losses: np.ndarray,
    risk: str,
    cvar_alpha: float,
) -> np.ndarray:
    pair_losses = np.asarray(pair_losses, dtype=np.float64)
    if pair_losses.ndim < 1 or pair_losses.shape[-1] < 1:
        raise ValueError("pair_losses must have a non-empty final dimension.")
    risk = risk.lower()
    if risk == "mean":
        return pair_losses.mean(axis=-1)
    if risk == "max":
        return pair_losses.max(axis=-1)
    if risk == "cvar":
        if not 0.0 <= cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must lie in [0, 1).")
        tail_count = max(
            int(math.ceil((1.0 - cvar_alpha) * pair_losses.shape[-1])),
            1,
        )
        partitioned = np.partition(
            pair_losses,
            pair_losses.shape[-1] - tail_count,
            axis=-1,
        )
        return partitioned[..., -tail_count:].mean(axis=-1)
    raise ValueError(f"Unknown pair risk: {risk}")


def audit_pair_loss_matrix(
    pair_loss_matrix: np.ndarray,
    sample_size: int,
    risk: str,
    cvar_alpha: float,
    trials: int,
    seed: int,
) -> dict[str, float | int | str | bool]:
    pair_loss_matrix = np.asarray(pair_loss_matrix, dtype=np.float64)
    if pair_loss_matrix.ndim != 2 or pair_loss_matrix.shape[1] < 2:
        raise ValueError("pair_loss_matrix must have shape [examples, pairs] with pairs >= 2.")
    if not np.isfinite(pair_loss_matrix).all():
        raise ValueError("pair_loss_matrix contains non-finite values.")
    pair_count = int(pair_loss_matrix.shape[1])
    sample_size = int(sample_size)
    trials = int(trials)
    if not 1 <= sample_size <= pair_count:
        raise ValueError("sample_size must lie between 1 and the full pair count.")
    if trials <= 0:
        raise ValueError("trials must be positive.")

    full = aggregate_pair_risk(pair_loss_matrix, risk, cvar_alpha)
    rng = np.random.default_rng(seed)
    sampled_indices = np.empty(
        (pair_loss_matrix.shape[0], trials, sample_size),
        dtype=np.int64,
    )
    for example in range(pair_loss_matrix.shape[0]):
        for trial in range(trials):
            sampled_indices[example, trial] = rng.choice(
                pair_count,
                size=sample_size,
                replace=False,
            )
    sampled_losses = np.take_along_axis(
        pair_loss_matrix[:, None, :],
        sampled_indices,
        axis=2,
    )
    approximations = aggregate_pair_risk(sampled_losses, risk, cvar_alpha)
    errors = approximations - full[:, None]
    expected_approximation = approximations.mean(axis=1)
    if pair_loss_matrix.shape[0] < 3:
        expected_rank_correlation = float("nan")
        trial_rank_correlations = np.asarray([], dtype=np.float64)
    elif np.allclose(full, full[0]) and np.allclose(
        expected_approximation, expected_approximation[0]
    ):
        expected_rank_correlation = 1.0
        trial_rank_correlations = np.ones(trials, dtype=np.float64)
    else:
        expected_rank_correlation = float(
            spearmanr(full, expected_approximation).statistic
        )
        trial_rank_correlations = np.asarray(
            [
                spearmanr(full, approximations[:, trial]).statistic
                for trial in range(trials)
            ],
            dtype=np.float64,
        )
        trial_rank_correlations = trial_rank_correlations[
            np.isfinite(trial_rank_correlations)
        ]

    full_scale = float(np.mean(np.abs(full)))
    bias = float(errors.mean())
    if risk == "max":
        full_tail_count = 1
    elif risk == "cvar":
        full_tail_count = max(
            int(math.ceil((1.0 - cvar_alpha) * pair_count)),
            1,
        )
    else:
        full_tail_count = pair_count
    top_full = np.argpartition(
        pair_loss_matrix,
        pair_count - full_tail_count,
        axis=1,
    )[:, -full_tail_count:]
    coverage = np.empty((pair_loss_matrix.shape[0], trials), dtype=np.float64)
    for example in range(pair_loss_matrix.shape[0]):
        coverage[example] = np.isin(
            sampled_indices[example],
            top_full[example],
        ).mean(axis=1)

    return {
        "risk": risk,
        "sample_size": sample_size,
        "pair_count": pair_count,
        "examples": int(pair_loss_matrix.shape[0]),
        "trials": trials,
        "full_risk_mean": float(full.mean()),
        "sampled_risk_mean": float(approximations.mean()),
        "bias": bias,
        "relative_bias": bias / max(full_scale, 1e-12),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "rank_correlation": (
            float(trial_rank_correlations.mean())
            if trial_rank_correlations.size
            else float("nan")
        ),
        "rank_correlation_q05": (
            float(np.quantile(trial_rank_correlations, 0.05))
            if trial_rank_correlations.size
            else float("nan")
        ),
        "expected_rank_correlation": expected_rank_correlation,
        "full_tail_pair_coverage": float(coverage.mean()),
    }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_positive_ints(value: str, label: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{label} must contain comma-separated integers.") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"{label} must contain positive integers.")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{label} contains duplicates.")
    return parsed


def _parse_risks(value: str) -> list[str]:
    parsed = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(parsed) - {"mean", "max", "cvar"})
    if not parsed or unknown:
        raise ValueError(
            "risks must contain mean, max, and/or cvar; "
            f"unknown values: {unknown}"
        )
    if len(set(parsed)) != len(parsed):
        raise ValueError("risks contains duplicates.")
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Measure the bias and ranking fidelity of sampled reference-pair "
            "training risks against all-pair risks on a frozen checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model", default="resnet50_imagenet")
    parser.add_argument("--reference_bank", default="")
    parser.add_argument(
        "--baseline_modes",
        default="zero,blur,noise,uniform,midpoint,bank",
    )
    parser.add_argument("--reference_bank_samples", type=int, default=4)
    parser.add_argument("--sample_sizes", default="4,8")
    parser.add_argument("--risks", default="mean,max,cvar")
    parser.add_argument("--cvar_alpha", type=float, default=0.75)
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--ig_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_mode", choices=["pred", "truth"], default="pred")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--out", default="outputs/pair_sampling_audit")
    parser.add_argument("--max_abs_relative_bias", type=float, default=0.05)
    parser.add_argument("--min_rank_correlation", type=float, default=0.90)
    parser.add_argument("--fail_on_threshold", action="store_true")
    args = parser.parse_args()

    try:
        sample_sizes = _parse_positive_ints(args.sample_sizes, "sample_sizes")
        risks = _parse_risks(args.risks)
    except ValueError as exc:
        parser.error(str(exc))
    if args.examples <= 0 or args.trials <= 0 or args.ig_steps <= 0:
        parser.error("--examples, --trials, and --ig_steps must be positive.")
    if args.reference_bank_samples < 2:
        parser.error("--reference_bank_samples must be at least 2.")
    if not 0.0 <= args.cvar_alpha < 1.0:
        parser.error("--cvar_alpha must lie in [0, 1).")
    if args.max_abs_relative_bias < 0.0:
        parser.error("--max_abs_relative_bias cannot be negative.")
    if not -1.0 <= args.min_rank_correlation <= 1.0:
        parser.error("--min_rank_correlation must lie in [-1, 1].")
    if "bank" in {
        mode.strip().lower()
        for mode in args.baseline_modes.split(",")
        if mode.strip()
    } and not args.reference_bank:
        parser.error("baseline mode 'bank' requires --reference_bank.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_profile = (
        "imagenet" if uses_imagenet_preprocessing(args.model) else "native"
    )
    _, test_set, _, _, _, meta = get_dataset(
        args.dataset,
        data_root=args.data_root,
        input_profile=input_profile,
    )
    model = get_model(
        args.model,
        args.dataset,
        num_classes=int(meta["num_classes"]),
    ).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    reference_bank = None
    if args.reference_bank:
        reference_bank, metadata = load_reference_bank_bundle(args.reference_bank)
        if metadata.get("dataset", args.dataset.lower()) != args.dataset.lower():
            parser.error("Reference bank was built for a different dataset.")
        if metadata.get("input_profile", input_profile) != input_profile:
            parser.error("Reference bank uses a different input profile.")
        reference_bank = reference_bank.to(device)

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(
        len(test_set),
        size=min(args.examples, len(test_set)),
        replace=False,
    )
    pair_rows = []
    pair_names = None
    for position, index in enumerate(indices, start=1):
        x, y = test_set[int(index)]
        x = x.to(device).unsqueeze(0)
        with torch.no_grad():
            predicted = int(model(x).argmax(dim=1).item())
        target = predicted if args.target_mode == "pred" else int(y)
        family = _baseline_family(
            x,
            args.baseline_modes,
            meta["baseline_min"],
            meta["baseline_max"],
            generator=_torch_generator(
                x,
                args.seed,
                "pair_sampling_audit",
                int(index),
            ),
            reference_bank=reference_bank,
            reference_bank_samples=args.reference_bank_samples,
        )
        names = list(family)
        active_pair_names = [
            (names[left], names[right])
            for left in range(len(names))
            for right in range(left + 1, len(names))
        ]
        if pair_names is None:
            pair_names = active_pair_names
        elif pair_names != active_pair_names:
            raise RuntimeError("Reference-pair ordering changed across audit examples.")
        attrs = {
            name: integrated_gradients(
                model,
                x,
                target,
                baseline,
                steps=args.ig_steps,
            )
            for name, baseline in family.items()
        }
        pair_rows.append(
            [
                float(
                    attribution_allocation_distance(
                        attrs[left],
                        attrs[right],
                    )[0].item()
                )
                for left, right in active_pair_names
            ]
        )
        print(f"[pair audit] {position}/{len(indices)}")

    pair_loss_matrix = np.asarray(pair_rows, dtype=np.float64)
    pair_count = int(pair_loss_matrix.shape[1])
    invalid_sizes = [size for size in sample_sizes if size >= pair_count]
    if invalid_sizes:
        parser.error(
            "Sample sizes must be smaller than the all-pair count "
            f"({pair_count}); invalid: {invalid_sizes}"
        )

    rows = []
    for risk_index, risk in enumerate(risks):
        for sample_size in sample_sizes:
            row = audit_pair_loss_matrix(
                pair_loss_matrix,
                sample_size=sample_size,
                risk=risk,
                cvar_alpha=args.cvar_alpha,
                trials=args.trials,
                seed=args.seed + 1009 * risk_index + sample_size,
            )
            row["passes_bias"] = (
                abs(float(row["relative_bias"]))
                <= args.max_abs_relative_bias
            )
            row["passes_rank"] = (
                np.isfinite(float(row["rank_correlation"]))
                and float(row["rank_correlation"])
                >= args.min_rank_correlation
            )
            row["passes"] = bool(row["passes_bias"] and row["passes_rank"])
            rows.append(row)

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "pair_sampling_fidelity.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "reference_bank": (
            os.path.abspath(args.reference_bank) if args.reference_bank else ""
        ),
        "reference_bank_sha256": (
            _sha256(args.reference_bank) if args.reference_bank else ""
        ),
        "dataset": args.dataset,
        "model": args.model,
        "input_profile": input_profile,
        "baseline_modes": args.baseline_modes,
        "reference_bank_samples": args.reference_bank_samples,
        "pair_names": pair_names,
        "test_indices": [int(index) for index in indices],
        "arguments": vars(args),
    }
    with open(
        os.path.join(args.out, "pair_sampling_fidelity.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote: {csv_path}")
    if args.fail_on_threshold and not all(bool(row["passes"]) for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
