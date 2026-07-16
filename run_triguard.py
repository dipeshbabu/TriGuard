import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from functools import lru_cache

import numpy as np
import torch
from torch import optim

from triguard.data import get_loaders
from triguard.eval import (
    accuracy,
    autoattack_accuracy,
    evaluate_certification,
    evaluate_appendix_metrics,
    evaluate_faithfulness,
    evaluate_main_metrics,
    pgd_accuracy,
)
from triguard.models import get_model
from triguard.models import split_model_name, uses_imagenet_preprocessing
from triguard.plots import save_curve_plot
from triguard.protocol import PROTOCOL_VERSION
from triguard.references import (
    load_index_reservation,
    load_reference_bank_bundle,
    reference_source_indices,
)
from triguard.results import append_csv, csv_contains_identity
from triguard.train import train_one_epoch


WORKSHOP_2026_GRID = {
    "mnist": ["simplecnn", "resnet50"],
    "fashionmnist": ["simplecnn", "resnet50"],
    "cifar10": ["resnet50", "densenet121", "vit_b_16"],
    "cifar100": ["resnet50", "densenet121", "vit_b_16"],
}

PRETRAINED_GRID = {
    "cifar10": [
        "resnet50_imagenet",
        "densenet121_imagenet",
        "vit_b_16_imagenet",
        "convnext_tiny_imagenet",
        "swin_t_imagenet",
    ],
    "cifar100": [
        "resnet50_imagenet",
        "densenet121_imagenet",
        "vit_b_16_imagenet",
        "convnext_tiny_imagenet",
        "swin_t_imagenet",
    ],
}

DATASET_ORDER = ["mnist", "fashionmnist", "cifar10", "cifar100"]
MODEL_ORDER = [
    "simplecnn",
    "resnet50",
    "densenet121",
    "vit_b_16",
    "resnet50_imagenet",
    "densenet121_imagenet",
    "vit_b_16_imagenet",
    "convnext_tiny_imagenet",
    "swin_t_imagenet",
]
DEFAULT_EPOCHS = {
    "mnist": 5,
    "fashionmnist": 5,
    "cifar10": 10,
    "cifar100": 15,
}
PRETRAINED_DEFAULT_EPOCHS = {
    "cifar10": 20,
    "cifar100": 30,
}
CSV_IDENTITY_FIELDS = [
    "protocol_version",
    "code_hash",
    "config_hash",
    "condition_hash",
    "comparison_hash",
]
TREATMENT_ARGUMENTS = {
    "lambda_entropy",
    "lambda_wads",
    "lambda_rar",
    "lambda_far",
    "lambda_curvature",
    "lambda_robust",
    "lambda_attr_mass",
    "reference_risk",
}
REGULARIZER_RESULT_COLUMNS = [
    "lambda_wads",
    "lambda_rar",
    "lambda_far",
    "lambda_curvature",
    "lambda_robust",
    "lambda_attr_mass",
    "reference_risk",
    "reference_cvar_alpha",
    "reference_distance",
    "reference_bank_samples",
    "eval_reference_bank_samples",
    "training_reservation_hash",
    "reference_bank_hash",
    "heldout_reference_bank_hash",
]
FAITHFULNESS_METRIC_COLUMNS = [
    "ig_del_auc_mean",
    "ig_ins_auc_mean",
    "sg2_del_auc_mean",
    "sg2_ins_auc_mean",
    "random_del_auc_mean",
    "random_ins_auc_mean",
    "ig_valid_n",
    "sg2_valid_n",
    "random_valid_n",
]
MAIN_METRIC_COLUMNS = [
    "attack_suite",
    "attack_eval_n",
    "empirical_probe_rate",
    "empirical_probe_violation_rate",
    "empirical_probe_min_margin_mean",
    "crown_rate",
    "crown_proven_rate",
    "crown_conditional_rate",
    "crown_attempted_n",
    "crown_certified_n",
    "entropy_mean",
    "entropy_raw_mean",
    "ads_mean",
    "ads_raw_l2_mean",
    "ads_raw_rms_mean",
    "ads_orthogonal_rms_mean",
    "ads_output_gap_mean",
    "wads_mean",
    "raw_wads_l2_mean",
    "raw_wads_rms_mean",
    "orthogonal_wads_rms_mean",
    "baseline_output_gap_mean",
    "baseline_attr_mass_min_mean",
    "baseline_attr_mass_q05",
    "baseline_attr_mass_near_zero_rate",
    "baseline_attr_mass_ratio_min_mean",
    "baseline_attr_mass_ratio_q05",
    "baseline_attr_mass_ratio_below_floor_rate",
    "heldout_wads_mean",
    "heldout_orthogonal_wads_rms_mean",
    "heldout_attr_mass_min_mean",
    "heldout_attr_mass_ratio_min_mean",
    "heldout_attr_mass_ratio_q05",
    "heldout_attr_mass_ratio_below_floor_rate",
    "heldout_reference_valid_n",
    "ig_completeness_error_mean",
    "ig_completeness_relative_error_mean",
    "ig_completeness_abs_max_mean",
    "ig_completeness_relative_max_mean",
    "pp_stability_l2_mean",
    "pp_stability_rms_mean",
    "pp_stability_cosine_mean",
    "pp_stability_topk_jaccard_mean",
    "pp_stability_keep_rate",
    "entropy_valid_n",
    "ads_valid_n",
    "wads_valid_n",
    "pp_stability_valid_n",
    "crown_valid_n",
    "crown_error_n",
    "crown_error_types",
    "faithfulness_enabled",
    "faithfulness_baseline",
    *FAITHFULNESS_METRIC_COLUMNS,
]


def validate_result_header(header, required=()):
    duplicates = sorted(
        {column for column in header if header.count(column) > 1}
    )
    if duplicates:
        raise ValueError(f"Duplicate result columns: {duplicates}")
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"Result schema is missing required columns: {missing}")


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def research_code_hash() -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    files = [
        os.path.join(root, "run_triguard.py"),
        os.path.join(root, "pyproject.toml"),
        os.path.join(root, "requirements.txt"),
        os.path.join(root, "requirements-autoattack.txt"),
        os.path.join(root, "requirements-dev.txt"),
        os.path.join(root, ".python-version"),
        os.path.join(root, "scripts", "00_install.sh"),
        os.path.join(root, "auto_LiRPA", "setup.py"),
    ]
    for source_root in [
        os.path.join(root, "triguard"),
        os.path.join(root, "auto_LiRPA", "auto_LiRPA"),
    ]:
        for directory, _, names in os.walk(source_root):
            files.extend(
                os.path.join(directory, name)
                for name in names
                if name.endswith(".py")
            )

    digest = hashlib.sha256()
    for path in sorted(set(files)):
        relative = os.path.relpath(path, root).replace(os.sep, "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def experiment_identity(args, **setting):
    excluded = {
        "out",
        "seed",
        "seeds",
        "dataset",
        "model",
        "mode",
        "save_ckpt",
        "checkpoint_path",
    }
    effective_args = {
        key: value for key, value in vars(args).items() if key not in excluded
    }
    artifact_hashes = {
        name: file_sha256(path)
        for name, path in {
            "load_ckpt": getattr(args, "load_ckpt", ""),
            "reference_bank": getattr(args, "reference_bank", ""),
            "heldout_reference_bank": getattr(
                args, "heldout_reference_bank", ""
            ),
            "exclude_train_indices_file": getattr(
                args, "exclude_train_indices_file", ""
            ),
        }.items()
        if path
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "code_hash": research_code_hash(),
        "arguments": effective_args,
        "artifact_hashes": artifact_hashes,
        "setting": setting,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    condition_payload = {
        **payload,
        "setting": {key: value for key, value in setting.items() if key != "seed"},
    }
    condition_canonical = json.dumps(
        condition_payload, sort_keys=True, separators=(",", ":")
    )
    condition_hash = hashlib.sha256(condition_canonical.encode("utf-8")).hexdigest()[:16]
    comparison_payload = {
        **condition_payload,
        "arguments": {
            key: value
            for key, value in effective_args.items()
            if key not in TREATMENT_ARGUMENTS
        },
        "setting": {
            key: value
            for key, value in condition_payload["setting"].items()
            if key not in TREATMENT_ARGUMENTS
        },
    }
    comparison_canonical = json.dumps(
        comparison_payload, sort_keys=True, separators=(",", ":")
    )
    comparison_hash = hashlib.sha256(comparison_canonical.encode("utf-8")).hexdigest()[:16]
    payload["condition_hash"] = condition_hash
    payload["comparison_hash"] = comparison_hash
    payload["execution"] = {
        "out": args.out,
        "save_ckpt": args.save_ckpt,
        "checkpoint_path": getattr(args, "checkpoint_path", ""),
    }
    return config_hash, payload


def write_run_manifest(out_dir: str, config_hash: str, payload: dict):
    manifest_dir = os.path.join(out_dir, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    path = os.path.join(manifest_dir, f"{config_hash}.json")
    git_status = _git_output("status", "--porcelain")
    manifest = {
        **payload,
        "config_hash": config_hash,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": _package_version("torchvision"),
            "numpy": np.__version__,
            "pandas": _package_version("pandas"),
            "scipy": _package_version("scipy"),
            "auto_lirpa": _package_version("auto-LiRPA"),
            "autoattack": _package_version("autoattack"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_dirty": bool(git_status) if git_status is not None else None,
        },
    }
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if any(
            existing.get(key) != manifest.get(key)
            for key in ["arguments", "artifact_hashes", "setting"]
        ):
            raise ValueError(f"Manifest hash collision at {path}")
        return
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def prepare_experiment_identity(args, **setting):
    config_hash, payload = experiment_identity(args, **setting)
    write_run_manifest(args.out, config_hash, payload)
    return config_hash, payload


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_compile(
    model: torch.nn.Module,
    device: torch.device,
    model_name: str,
    enable_compile: bool = False,
):
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        # torch.compile can improve throughput, but it has been less reliable
        # than eager mode across our benchmark settings.
        base_model_name, _ = split_model_name(model_name)
        if enable_compile and hasattr(torch, "compile") and base_model_name not in {"vit_b_16", "vit"}:
            model = torch.compile(model)
    return model


def parse_seeds(seed: int, seeds_arg: str | None):
    if not seeds_arg:
        return [seed]
    parsed = [int(x.strip()) for x in seeds_arg.split(",") if x.strip()]
    if not parsed:
        raise ValueError("--seeds did not contain any integers.")
    if len(set(parsed)) != len(parsed):
        raise ValueError("--seeds contains duplicate values.")
    return parsed


def scale_eps(eps, scale: float):
    if isinstance(eps, (list, tuple)):
        return tuple(float(v) * scale for v in eps)
    return float(eps) * scale


def resolve_epochs(args, dataset: str, model_name: str) -> int:
    if args.epochs is not None:
        return args.epochs
    if uses_imagenet_preprocessing(model_name):
        return PRETRAINED_DEFAULT_EPOCHS.get(dataset.lower(), 20)
    return DEFAULT_EPOCHS.get(dataset.lower(), 5)


def resolve_cert_eps(args, attack_eps, pixel_eps: float):
    if args.cert_eps is not None:
        return float(args.cert_eps)
    if args.cert_pixel_eps is not None:
        return scale_eps(attack_eps, args.cert_pixel_eps / pixel_eps)
    return scale_eps(attack_eps, args.cert_eps_scale)


@lru_cache(maxsize=None)
def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def regularizer_fields(args):
    return {
        "lambda_wads": args.lambda_wads,
        "lambda_rar": args.lambda_rar,
        "lambda_far": args.lambda_far,
        "lambda_curvature": args.lambda_curvature,
        "lambda_robust": args.lambda_robust,
        "lambda_attr_mass": args.lambda_attr_mass,
        "reference_risk": args.reference_risk,
        "reference_cvar_alpha": args.reference_cvar_alpha,
        "reference_distance": args.reference_distance,
        "reference_bank_samples": args.reference_bank_samples,
        "eval_reference_bank_samples": args.eval_reference_bank_samples,
        "training_reservation_hash": (
            file_sha256(args.exclude_train_indices_file)
            if args.exclude_train_indices_file
            else ""
        ),
        "reference_bank_hash": (
            file_sha256(args.reference_bank) if args.reference_bank else ""
        ),
        "heldout_reference_bank_hash": (
            file_sha256(args.heldout_reference_bank)
            if args.heldout_reference_bank
            else ""
        ),
    }


def checkpoint_regularizer_suffix(args) -> str:
    if (
        args.lambda_wads == 0
        and args.lambda_rar == 0
        and args.lambda_far == 0
        and args.lambda_curvature == 0
        and args.lambda_robust == 0
        and args.lambda_attr_mass == 0
    ):
        return ""
    return (
        f"_wads{args.lambda_wads:.3f}"
        f"_rar{args.lambda_rar:.3f}"
        f"_far{args.lambda_far:.3f}"
        f"_curv{args.lambda_curvature:.3f}"
        f"_rob{args.lambda_robust:.3f}"
        f"_mass{args.lambda_attr_mass:.3f}"
        f"_risk{args.reference_risk}"
        f"_dist{args.reference_distance}"
        f"_bank{file_sha256(args.reference_bank) if args.reference_bank else 'none'}"
    )


def resolve_checkpoint_path(
    args,
    *,
    mode: str,
    dataset: str,
    model_name: str,
    seed: int,
    lam: float,
    config_hash: str,
) -> str | None:
    if not args.save_ckpt:
        return None
    if args.checkpoint_path:
        return args.checkpoint_path
    return os.path.join(
        args.out,
        "checkpoints",
        f"{mode}_{dataset}_{model_name}_seed{seed}_lam{lam:.3f}"
        f"{checkpoint_regularizer_suffix(args)}"
        f"_cfg{config_hash}.pt",
    )


def checkpoint_artifacts_complete(
    args,
    *,
    mode: str,
    dataset: str,
    model_name: str,
    seed: int,
    lam: float,
    config_hash: str,
) -> bool:
    if not args.save_ckpt or args.eval_only:
        return True
    path = resolve_checkpoint_path(
        args,
        mode=mode,
        dataset=dataset,
        model_name=model_name,
        seed=seed,
        lam=lam,
        config_hash=config_hash,
    )
    if not path or not os.path.isfile(path):
        return False
    sidecar = f"{path}.meta.json"
    if not os.path.isfile(sidecar):
        return False
    try:
        with open(sidecar, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        metadata.get("config_hash") == config_hash
        and metadata.get("code_hash") == research_code_hash()
        and metadata.get("checkpoint_sha256")
        and os.path.abspath(metadata.get("checkpoint", "")) == os.path.abspath(path)
    )


def _full_file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checkpoint_metadata(path: str, metadata: dict):
    sidecar = f"{path}.meta.json"
    payload = {
        **metadata,
        "checkpoint": os.path.abspath(path),
        "checkpoint_sha256": _full_file_sha256(path),
    }
    temporary = f"{sidecar}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, sidecar)


def build_training_setup(args, dataset: str, model_name: str, model, epochs: int):
    dataset = dataset.lower()
    model_name = model_name.lower()

    if uses_imagenet_preprocessing(model_name):
        lr = args.lr if args.lr is not None else 5e-5
        opt = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=args.weight_decay,
        )
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(epochs - args.warmup_epochs, 1)
        )
        if args.warmup_epochs > 0 and epochs > args.warmup_epochs:
            scheduler = optim.lr_scheduler.SequentialLR(
                opt,
                schedulers=[
                    optim.lr_scheduler.LinearLR(
                        opt, start_factor=0.1, total_iters=args.warmup_epochs
                    ),
                    main_scheduler,
                ],
                milestones=[args.warmup_epochs],
            )
        else:
            scheduler = main_scheduler
        grad_clip = args.grad_clip if args.grad_clip is not None else 1.0
        return {
            "optimizer": opt,
            "scheduler": scheduler,
            "grad_clip": grad_clip,
            "optimizer_name": "adamw_imagenet",
            "lr": lr,
        }

    if dataset in {"cifar10", "cifar100"} and model_name in {"resnet50", "densenet121"}:
        lr = args.lr if args.lr is not None else 0.05
        opt = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        grad_clip = args.grad_clip if args.grad_clip is not None else 1.0
        return {
            "optimizer": opt,
            "scheduler": scheduler,
            "grad_clip": grad_clip,
            "optimizer_name": "sgd",
            "lr": lr,
        }

    if dataset in {"cifar10", "cifar100"} and model_name in {"vit_b_16", "vit"}:
        lr = args.lr if args.lr is not None else 3e-4
        opt = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.05,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        grad_clip = args.grad_clip if args.grad_clip is not None else 1.0
        return {
            "optimizer": opt,
            "scheduler": scheduler,
            "grad_clip": grad_clip,
            "optimizer_name": "adamw",
            "lr": lr,
        }

    lr = args.lr if args.lr is not None else 1e-3
    opt = optim.Adam(model.parameters(), lr=lr)
    return {
        "optimizer": opt,
        "scheduler": None,
        "grad_clip": args.grad_clip,
        "optimizer_name": "adam",
        "lr": lr,
    }


def train_model(
    model,
    train_loader,
    opt,
    device,
    epochs: int,
    lambda_entropy: float,
    scaler=None,
    entropy_model=None,
    scheduler=None,
    grad_clip: float | None = None,
    patience: int = 3,
    min_delta: float = 1e-4,
    ckpt_path: str | None = None,
    state_model=None,
    train_kwargs=None,
    selection_policy: str = "fixed",
):
    if selection_policy not in {"fixed", "train_loss"}:
        raise ValueError(f"Unknown selection policy: {selection_policy}")
    best = float("inf")
    bad = 0
    best_state = None
    state_model = state_model or model
    train_kwargs = train_kwargs or {}

    def cpu_state_dict():
        return {
            key: value.detach().cpu().clone()
            for key, value in state_model.state_dict().items()
        }

    def validate_finite_state(state):
        if any(
            tensor.is_floating_point() and not torch.isfinite(tensor).all()
            for tensor in state.values()
        ):
            raise FloatingPointError(
                "Training produced non-finite model parameters or buffers."
            )

    for _ in range(epochs):
        loss = train_one_epoch(
            model,
            train_loader,
            opt,
            device,
            lambda_entropy=lambda_entropy,
            scaler=scaler,
            entropy_model=entropy_model,
            grad_clip=grad_clip,
            **train_kwargs,
        )
        if scheduler is not None:
            scheduler.step()
        loss = float(loss)
        if not np.isfinite(loss):
            raise FloatingPointError(
                "Training produced a non-finite loss; checkpoint and results were not written."
            )
        if selection_policy == "fixed":
            best = loss
            continue
        if loss < best - min_delta:
            best = loss
            bad = 0
            best_state = cpu_state_dict()
        else:
            bad += 1
            if bad >= patience:
                break

    if selection_policy == "fixed":
        final_state = cpu_state_dict()
        validate_finite_state(final_state)
        if ckpt_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
            torch.save(final_state, ckpt_path)
    elif best_state is not None:
        validate_finite_state(best_state)
        state_model.load_state_dict(best_state)
        if ckpt_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
            torch.save(best_state, ckpt_path)
    return best


def merged_grid(grid_name: str):
    grid_name = grid_name.lower()
    if grid_name == "workshop":
        return WORKSHOP_2026_GRID
    if grid_name == "pretrained":
        return PRETRAINED_GRID
    if grid_name == "all":
        merged = {ds: list(models) for ds, models in WORKSHOP_2026_GRID.items()}
        for ds, models in PRETRAINED_GRID.items():
            merged.setdefault(ds, [])
            for model in models:
                if model not in merged[ds]:
                    merged[ds].append(model)
        return merged
    raise ValueError(f"Unknown grid: {grid_name}")


def get_experiment_grid(dataset: str | None, model: str | None, grid_name: str):
    grid = merged_grid(grid_name)
    if dataset and model:
        ds = dataset.lower()
        model_name = model.lower()
        all_grid = merged_grid("all")
        if ds not in all_grid or model_name not in all_grid[ds]:
            raise ValueError(
                f"Unsupported explicit dataset/model pair: {ds}/{model_name}."
            )
        return [(ds, model_name)]

    if dataset:
        ds = dataset.lower()
        if ds not in grid:
            raise ValueError(f"Unknown dataset: {ds}")
        return [(ds, m) for m in grid[ds]]

    if model:
        m = model.lower()
        pairs = []
        for ds in DATASET_ORDER:
            if ds in grid and m in grid[ds]:
                pairs.append((ds, m))
        if not pairs:
            raise ValueError(f"Model {m} is not used in the {grid_name} grid.")
        return pairs

    pairs = []
    for ds in DATASET_ORDER:
        for m in grid.get(ds, []):
            pairs.append((ds, m))
    return pairs


def run_one_setting(
    args,
    seed: int,
    ds: str,
    model_name: str,
    lam: float,
    device: torch.device,
    mode: str,
    cert_pixel_values: list[float] | None = None,
):
    set_seed(seed)
    config_hash, manifest_payload = prepare_experiment_identity(
        args,
        mode=mode,
        dataset=ds,
        model=model_name,
        seed=seed,
        lambda_entropy=lam,
    )
    identity = {
        "protocol_version": PROTOCOL_VERSION,
        "code_hash": manifest_payload["code_hash"],
        "config_hash": config_hash,
        "condition_hash": manifest_payload["condition_hash"],
        "comparison_hash": manifest_payload["comparison_hash"],
    }
    epochs = resolve_epochs(args, ds, model_name)
    if (
        uses_imagenet_preprocessing(model_name)
        and args.warmup_epochs > 0
        and args.warmup_epochs >= epochs
    ):
        raise ValueError(
            "For ImageNet-preprocessed models, --warmup_epochs must be smaller "
            "than the total epoch budget."
        )
    input_profile = "imagenet" if uses_imagenet_preprocessing(model_name) else "native"
    reference_bank = None
    heldout_reference_bank = None
    reserved_reference_indices = set()
    reservation_indices = set()
    bank_source_indices = {}
    if args.exclude_train_indices_file:
        reservation, reservation_metadata = load_index_reservation(
            args.exclude_train_indices_file
        )
        source_dataset = reservation_metadata.get("dataset")
        source_profile = reservation_metadata.get("input_profile")
        if source_dataset is not None and str(source_dataset).lower() != ds.lower():
            raise ValueError(
                "Training reservation was built for "
                f"{source_dataset}, not {ds}."
            )
        if source_profile is not None and source_profile != input_profile:
            raise ValueError(
                "Training reservation uses profile "
                f"{source_profile}, not {input_profile}."
            )
        reserved_reference_indices.update(reservation)
        reservation_indices.update(reservation)
    for path, heldout in [
        (args.reference_bank, False),
        (args.heldout_reference_bank, True),
    ]:
        if not path:
            continue
        bank, bank_metadata = load_reference_bank_bundle(path)
        source_dataset = bank_metadata.get("dataset")
        source_profile = bank_metadata.get("input_profile")
        if source_dataset is not None and str(source_dataset).lower() != ds.lower():
            raise ValueError(f"Reference bank {path} was built for {source_dataset}, not {ds}.")
        if source_profile is not None and source_profile != input_profile:
            raise ValueError(
                f"Reference bank {path} uses profile {source_profile}, not {input_profile}."
            )
        role = "heldout" if heldout else "train"
        source_indices = set(
            reference_source_indices(bank_metadata, bank.size(0), role)
        )
        bank_source_indices[role] = source_indices
        reserved_reference_indices.update(source_indices)
        if heldout:
            heldout_reference_bank = bank
        else:
            reference_bank = bank
    if {"train", "heldout"} <= set(bank_source_indices):
        overlap = bank_source_indices["train"] & bank_source_indices["heldout"]
        if overlap:
            raise ValueError(
                "Training and held-out reference banks share source indices."
            )
    if reservation_indices:
        outside_reservation = set().union(*bank_source_indices.values()) - reservation_indices
        if outside_reservation:
            raise ValueError(
                "Reference-bank images are not all members of the supplied "
                "training reservation."
            )
    train_loader, test_loader, test_set, clamp_min, clamp_max, eps, meta = get_loaders(
        ds,
        args.batch,
        num_workers=args.num_workers,
        data_root=args.data_root,
        input_profile=input_profile,
        exclude_train_indices=reserved_reference_indices,
        seed=seed,
    )

    num_classes = int(meta["num_classes"])
    baseline_min = meta["baseline_min"]
    baseline_max = meta["baseline_max"]
    pixel_eps = float(meta["pixel_eps"])

    base_model = get_model(model_name, ds, num_classes=num_classes).to(device)
    model = maybe_compile(base_model, device, model_name, enable_compile=args.compile)
    entropy_model = base_model if model is not base_model else model
    eval_model = base_model if model is not base_model else model
    training_setup = build_training_setup(args, ds, model_name, model, epochs)
    opt = training_setup["optimizer"]
    scheduler = training_setup["scheduler"]
    grad_clip = training_setup["grad_clip"]

    print(
        f"[Train Config] {ds}/{model_name}: "
        f"epochs={epochs}, input_profile={input_profile}, optimizer={training_setup['optimizer_name']}, "
        f"lr={training_setup['lr']}, scheduler={'cosine' if scheduler is not None else 'none'}, "
        f"grad_clip={grad_clip}, compile={args.compile}, "
        f"wads={args.lambda_wads}, rar={args.lambda_rar}, far={args.lambda_far}, "
        f"curvature={args.lambda_curvature}, robust={args.lambda_robust}"
        f", attr_mass={args.lambda_attr_mass}, reference_risk={args.reference_risk}, "
        f"reference_distance={args.reference_distance}"
    )

    ckpt_path = resolve_checkpoint_path(
        args,
        mode=mode,
        dataset=ds,
        model_name=model_name,
        seed=seed,
        lam=lam,
        config_hash=config_hash,
    )

    if args.load_ckpt:
        state = torch.load(args.load_ckpt, map_location=device, weights_only=True)
        eval_model.load_state_dict(state)

    train_seconds = 0.0
    train_start = time.perf_counter()
    if not args.eval_only:
        train_model(
            model,
            train_loader,
            opt,
            device,
            epochs=epochs,
            lambda_entropy=lam,
            scaler=(torch.amp.GradScaler("cuda")
                    if device.type == "cuda" else None),
            entropy_model=entropy_model,
            scheduler=scheduler,
            grad_clip=grad_clip,
            patience=args.patience,
            min_delta=args.min_delta,
            ckpt_path=ckpt_path,
            state_model=eval_model,
            selection_policy=args.selection_policy,
            train_kwargs={
                "lambda_wads": args.lambda_wads,
                "lambda_rar": args.lambda_rar,
                "lambda_far": args.lambda_far,
                "lambda_curvature": args.lambda_curvature,
                "lambda_robust": args.lambda_robust,
                "lambda_attr_mass": args.lambda_attr_mass,
                "triguard_ig_steps": args.triguard_ig_steps,
                "baseline_modes": args.baseline_modes,
                "reference_risk": args.reference_risk,
                "reference_cvar_alpha": args.reference_cvar_alpha,
                "reference_pair_samples": args.reference_pair_samples,
                "reference_distance": args.reference_distance,
                "reference_bank": reference_bank,
                "reference_bank_samples": args.reference_bank_samples,
                "attr_mass_floor": args.attr_mass_floor,
                "attr_robust_baseline": args.attr_robust_baseline,
                "far_samples": args.far_samples,
                "baseline_min": baseline_min,
                "baseline_max": baseline_max,
                "curvature_noise_std": args.curvature_noise_std,
                "attr_robust_eps": scale_eps(eps, args.attr_robust_eps_scale),
                "attr_robust_alpha": scale_eps(eps, args.attr_robust_alpha_scale),
                "robust_eps": scale_eps(eps, args.robust_eps_scale),
                "robust_alpha": scale_eps(eps, args.robust_alpha_scale),
                "robust_clamp_min": clamp_min,
                "robust_clamp_max": clamp_max,
            },
        )
        train_seconds = time.perf_counter() - train_start
        if ckpt_path is not None:
            write_checkpoint_metadata(
                ckpt_path,
                {
                    **identity,
                    "dataset": ds,
                    "model": model_name,
                    "seed": seed,
                    "lambda_entropy": lam,
                    "epochs": epochs,
                    "selection_policy": args.selection_policy,
                    "training_reservation_sha256": (
                        _full_file_sha256(args.exclude_train_indices_file)
                        if args.exclude_train_indices_file
                        else ""
                    ),
                    **regularizer_fields(args),
                },
            )

    eval_model.eval()
    alpha = scale_eps(eps, 1.0 / 8.0)
    cert_eps = resolve_cert_eps(args, eps, pixel_eps)

    if mode in ["main", "lambda"]:
        eval_start = time.perf_counter()
        clean = accuracy(eval_model, test_loader, device)
        if args.attack_suite == "autoattack":
            adv_acc, attack_eval_n = autoattack_accuracy(
                eval_model,
                test_loader,
                device,
                pixel_eps=pixel_eps,
                normalization_mean=meta["normalization_mean"],
                normalization_std=meta["normalization_std"],
                max_samples=args.autoattack_samples,
                batch_size=args.autoattack_batch,
                seed=seed,
            )
        else:
            adv_acc = pgd_accuracy(
                eval_model,
                test_loader,
                device,
                eps,
                alpha,
                args.pgd_steps,
                clamp_min,
                clamp_max,
                max_batches=args.eval_batches_adv,
                restarts=args.pgd_restarts,
                seed=seed,
            )
            attack_eval_n = min(
                len(test_set), args.eval_batches_adv * test_loader.batch_size
            )
        adv_error = 1.0 - adv_acc
        main_metrics = evaluate_main_metrics(
            eval_model,
            test_set,
            device,
            eps=eps,
            alpha=alpha,
            pgd_steps=args.pgd_steps,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            ig_steps=args.ig_steps,
            k=args.K_attr,
            seed=seed,
            target_mode=args.target_mode,
            empirical_probe_samples=args.empirical_probe_samples,
            do_crown=(not args.skip_crown),
            cert_eps=cert_eps,
            baseline_min=baseline_min,
            baseline_max=baseline_max,
            baseline_modes=args.baseline_modes,
            stability_modes=args.stability_modes,
            stability_topk_fraction=args.stability_topk_fraction,
            stability_noise_std=args.stability_noise_std,
            reference_bank=reference_bank,
            heldout_reference_bank=heldout_reference_bank,
            reference_bank_samples=args.eval_reference_bank_samples,
            attr_mass_floor=args.attr_mass_floor,
        )
        if args.main_faithfulness:
            faithfulness = evaluate_faithfulness(
                eval_model,
                test_set,
                device,
                k=args.K_faith,
                ig_steps=args.ig_steps,
                delins_steps=args.delins_steps,
                seed=seed,
                baseline_mode=args.faithfulness_baseline,
                target_mode=args.target_mode,
                smoothgrad_noise=scale_eps(
                    eps, args.smoothgrad_noise_pixel / pixel_eps
                ),
                smoothgrad_samples=50,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
                baseline_min=baseline_min,
                baseline_max=baseline_max,
            )
            faithfulness_metrics = {
                column: faithfulness[column]
                for column in FAITHFULNESS_METRIC_COLUMNS
            }
        else:
            faithfulness_metrics = {
                column: (
                    0
                    if column.endswith("_valid_n")
                    else float("nan")
                )
                for column in FAITHFULNESS_METRIC_COLUMNS
            }
        eval_seconds = time.perf_counter() - eval_start
        return {
            **identity,
            "dataset": ds,
            "model": model_name,
            "seed": seed,
            "input_profile": input_profile,
            "lambda_entropy": lam,
            **regularizer_fields(args),
            "baseline_modes": args.baseline_modes,
            "attr_robust_baseline": args.attr_robust_baseline,
            "cert_eps": cert_eps,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            "total_seconds": train_seconds + eval_seconds,
            "clean_acc": clean,
            "adv_error": adv_error,
            "attack_suite": args.attack_suite,
            "attack_eval_n": attack_eval_n,
            **main_metrics,
            "faithfulness_enabled": int(args.main_faithfulness),
            "faithfulness_baseline": args.faithfulness_baseline,
            **faithfulness_metrics,
        }

    if mode == "cert_sweep":
        rows = []
        if cert_pixel_values is None:
            cert_pixel_values = [
                float(x.strip()) for x in args.cert_eps_list.split(",") if x.strip()
            ]
        for pixel_value in cert_pixel_values:
            active_cert_eps = scale_eps(eps, pixel_value / pixel_eps)
            cert_metrics = evaluate_certification(
                eval_model,
                test_set,
                device,
                cert_eps=active_cert_eps,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
                k=args.K_attr,
                seed=seed,
            )
            rows.append(
                {
                    **identity,
                    "dataset": ds,
                    "model": model_name,
                    "seed": seed,
                    "input_profile": input_profile,
                    "lambda_entropy": lam,
                    **regularizer_fields(args),
                    "cert_pixel_eps": pixel_value,
                    **cert_metrics,
                }
            )
        return rows

    if mode == "baseline":
        appendix = evaluate_appendix_metrics(
            eval_model,
            test_set,
            device,
            ig_steps=args.ig_steps,
            k=args.K_attr,
            seed=seed,
            target_mode=args.target_mode,
            baseline_min=baseline_min,
            baseline_max=baseline_max,
        )
        return {
            **identity,
            "dataset": ds,
            "model": model_name,
            "seed": seed,
            "input_profile": input_profile,
            "K": args.K_attr,
            **regularizer_fields(args),
            **{k: appendix[k] for k in appendix if k.startswith("ads_")},
        }

    if mode == "faithfulness":
        res = evaluate_faithfulness(
            eval_model,
            test_set,
            device,
            k=args.K_faith,
            ig_steps=args.ig_steps,
            delins_steps=args.delins_steps,
            seed=seed,
            baseline_mode=args.faithfulness_baseline,
            target_mode=args.target_mode,
            smoothgrad_noise=scale_eps(
                eps, args.smoothgrad_noise_pixel / pixel_eps
            ),
            smoothgrad_samples=50,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            baseline_min=baseline_min,
            baseline_max=baseline_max,
        )
        fig_dir = os.path.join(args.out, "figures",
                               "faithfulness", ds, model_name, f"seed_{seed}")
        for i, (tag, del_curve, ins_curve) in enumerate(res["curves"][:4]):
            save_curve_plot(
                del_curve,
                ins_curve,
                f"{ds}/{model_name} {tag}",
                os.path.join(fig_dir, f"{tag}_curve_{i}.png"),
            )
        return {
            **identity,
            "dataset": ds,
            "model": model_name,
            "seed": seed,
            "input_profile": input_profile,
            "K": args.K_faith,
            "faithfulness_baseline": args.faithfulness_baseline,
            **regularizer_fields(args),
            "ig_del_auc_mean": res["ig_del_auc_mean"],
            "ig_ins_auc_mean": res["ig_ins_auc_mean"],
            "sg2_del_auc_mean": res["sg2_del_auc_mean"],
            "sg2_ins_auc_mean": res["sg2_ins_auc_mean"],
            "random_del_auc_mean": res["random_del_auc_mean"],
            "random_ins_auc_mean": res["random_ins_auc_mean"],
            "ig_valid_n": res["ig_valid_n"],
            "sg2_valid_n": res["sg2_valid_n"],
            "random_valid_n": res["random_valid_n"],
        }

    raise ValueError(f"Unsupported mode: {mode}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs/icml2026")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=str, default="")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lambda_entropy", type=float, default=0.05)
    p.add_argument("--lambda_wads", type=float, default=0.0)
    p.add_argument("--lambda_rar", type=float, default=0.0)
    p.add_argument("--lambda_far", type=float, default=0.0)
    p.add_argument("--lambda_curvature", type=float, default=0.0)
    p.add_argument("--lambda_robust", type=float, default=0.0)
    p.add_argument("--lambda_attr_mass", type=float, default=0.0)
    p.add_argument("--triguard_ig_steps", type=int, default=8)
    p.add_argument("--baseline_modes", type=str, default="zero,blur,noise,uniform,midpoint")
    p.add_argument(
        "--reference_risk", choices=["mean", "cvar", "max"], default="max"
    )
    p.add_argument("--reference_cvar_alpha", type=float, default=0.75)
    p.add_argument("--reference_pair_samples", type=int, default=0)
    p.add_argument(
        "--reference_distance",
        choices=["allocation", "orthogonal_rms"],
        default="allocation",
    )
    p.add_argument("--reference_bank", type=str, default="")
    p.add_argument("--heldout_reference_bank", type=str, default="")
    p.add_argument("--exclude_train_indices_file", type=str, default="")
    p.add_argument("--reference_bank_samples", type=int, default=4)
    p.add_argument("--eval_reference_bank_samples", type=int, default=16)
    p.add_argument("--attr_mass_floor", type=float, default=0.9)
    p.add_argument("--attr_robust_baseline", type=str, default="zero")
    p.add_argument("--far_samples", type=int, default=2)
    p.add_argument("--curvature_noise_std", type=float, default=0.01)
    p.add_argument("--attr_robust_eps_scale", type=float, default=1.0)
    p.add_argument("--attr_robust_alpha_scale", type=float, default=0.25)
    p.add_argument("--robust_eps_scale", type=float, default=1.0)
    p.add_argument("--robust_alpha_scale", type=float, default=0.25)
    p.add_argument("--stability_modes", type=str, default="noise,brightness,contrast,blur,shift")
    p.add_argument("--stability_topk_fraction", type=float, default=0.05)
    p.add_argument("--stability_noise_std", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--ig_steps", type=int, default=50)
    p.add_argument("--pgd_steps", type=int, default=40)
    p.add_argument("--pgd_restarts", type=int, default=5)
    p.add_argument(
        "--attack_suite", choices=["pgd", "autoattack"], default="pgd"
    )
    p.add_argument("--autoattack_samples", type=int, default=1000)
    p.add_argument("--autoattack_batch", type=int, default=128)
    p.add_argument("--cert_eps", type=float, default=None)
    p.add_argument("--cert_pixel_eps", type=float, default=None)
    p.add_argument("--cert_eps_scale", type=float, default=1.0)
    p.add_argument("--cert_eps_list", type=str, default="0.0,0.0039215686,0.0078431373,0.0156862745,0.031372549")
    p.add_argument("--eval_batches_adv", type=int, default=10)
    p.add_argument("--K_attr", type=int, default=100)
    p.add_argument("--K_faith", type=int, default=50)
    p.add_argument("--delins_steps", type=int, default=50)
    p.add_argument("--smoothgrad_noise_pixel", type=float, default=0.05)
    p.add_argument(
        "--main_faithfulness",
        action="store_true",
        help="Evaluate faithfulness controls on the same checkpoint in main/lambda mode.",
    )
    p.add_argument(
        "--faithfulness_baseline",
        type=str,
        default="blur",
        choices=["zero", "blur", "noise", "uniform", "midpoint"],
    )
    p.add_argument("--target_mode", type=str,
                   default="pred", choices=["truth", "pred"])
    p.add_argument("--mode", type=str, default="main",
                   choices=["main", "lambda", "baseline", "faithfulness", "cert_sweep"])
    p.add_argument("--grid", type=str, default="workshop",
                   choices=["workshop", "pretrained", "all"])
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--lambda_list", type=str, default="0.0,0.01,0.05,0.1")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument(
        "--selection_policy",
        choices=["fixed", "train_loss"],
        default="fixed",
        help="Use fixed epochs for unbiased primary comparisons; train_loss is legacy.",
    )
    p.add_argument("--fast", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--skip_crown", action="store_true")
    p.add_argument("--save_ckpt", action="store_true")
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Optional exact save path; default checkpoint names include the run hash.",
    )
    p.add_argument("--load_ckpt", type=str, default="")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--empirical_probe_samples",
        "--bound_probe_samples",
        dest="empirical_probe_samples",
        type=int,
        default=16,
        help=(
            "Number of random perturbations used by the empirical prediction-"
            "preservation probe. The legacy --bound_probe_samples spelling is "
            "accepted for compatibility; this metric is not a certificate."
        ),
    )
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    nonfinite_arguments = [
        name
        for name, value in vars(args).items()
        if isinstance(value, float) and not np.isfinite(value)
    ]
    if nonfinite_arguments:
        p.error(
            "These floating-point arguments must be finite: "
            f"{', '.join(nonfinite_arguments)}"
        )
    positive_arguments = {
        "batch": args.batch,
        "triguard_ig_steps": args.triguard_ig_steps,
        "ig_steps": args.ig_steps,
        "pgd_steps": args.pgd_steps,
        "pgd_restarts": args.pgd_restarts,
        "autoattack_samples": args.autoattack_samples,
        "autoattack_batch": args.autoattack_batch,
        "eval_batches_adv": args.eval_batches_adv,
        "K_attr": args.K_attr,
        "K_faith": args.K_faith,
        "delins_steps": args.delins_steps,
        "empirical_probe_samples": args.empirical_probe_samples,
    }
    invalid_positive = [
        name for name, value in positive_arguments.items() if int(value) <= 0
    ]
    if invalid_positive:
        p.error(f"These arguments must be positive: {', '.join(invalid_positive)}")
    if args.epochs is not None and args.epochs <= 0:
        p.error("--epochs must be positive when specified.")
    if args.num_workers < 0:
        p.error("--num_workers cannot be negative.")
    if not 0.0 <= args.reference_cvar_alpha < 1.0:
        p.error("--reference_cvar_alpha must lie in [0, 1).")
    if args.reference_bank_samples < 2 or args.eval_reference_bank_samples < 2:
        p.error("Reference-bank sample counts must each be at least 2.")
    if args.reference_pair_samples < 0:
        p.error("--reference_pair_samples cannot be negative.")
    if args.far_samples <= 0:
        p.error("--far_samples must be positive.")
    if not 0.0 < args.stability_topk_fraction <= 1.0:
        p.error("--stability_topk_fraction must lie in (0, 1].")
    if args.attr_mass_floor < 0.0:
        p.error("--attr_mass_floor cannot be negative.")
    nonnegative_arguments = {
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "min_delta": args.min_delta,
        "curvature_noise_std": args.curvature_noise_std,
        "stability_noise_std": args.stability_noise_std,
        "smoothgrad_noise_pixel": args.smoothgrad_noise_pixel,
        "attr_robust_eps_scale": args.attr_robust_eps_scale,
        "attr_robust_alpha_scale": args.attr_robust_alpha_scale,
        "robust_eps_scale": args.robust_eps_scale,
        "robust_alpha_scale": args.robust_alpha_scale,
        "cert_eps_scale": args.cert_eps_scale,
    }
    invalid_nonnegative = [
        name for name, value in nonnegative_arguments.items() if value < 0.0
    ]
    if invalid_nonnegative:
        p.error(
            "These arguments cannot be negative: "
            f"{', '.join(invalid_nonnegative)}"
        )
    if args.grad_clip is not None and args.grad_clip <= 0.0:
        p.error("--grad_clip must be positive when specified.")
    if args.patience <= 0:
        p.error("--patience must be positive.")
    if args.cert_eps is not None and args.cert_eps < 0.0:
        p.error("--cert_eps cannot be negative.")
    if args.cert_pixel_eps is not None and args.cert_pixel_eps < 0.0:
        p.error("--cert_pixel_eps cannot be negative.")
    try:
        cert_pixel_values = [
            float(value.strip())
            for value in args.cert_eps_list.split(",")
            if value.strip()
        ]
    except ValueError:
        p.error("--cert_eps_list must contain comma-separated numbers.")
    if not cert_pixel_values or any(
        not np.isfinite(value) or value < 0.0 for value in cert_pixel_values
    ):
        p.error("--cert_eps_list must contain finite nonnegative radii.")
    if len(set(cert_pixel_values)) != len(cert_pixel_values):
        p.error("--cert_eps_list contains duplicate radii.")
    try:
        lambda_values = [
            float(value.strip())
            for value in args.lambda_list.split(",")
            if value.strip()
        ]
    except ValueError:
        p.error("--lambda_list must contain comma-separated numbers.")
    if not lambda_values or any(
        not np.isfinite(value) or value < 0.0 for value in lambda_values
    ):
        p.error("--lambda_list must contain finite nonnegative weights.")
    if len(set(lambda_values)) != len(lambda_values):
        p.error("--lambda_list contains duplicate weights.")
    lambda_names = [
        "lambda_entropy",
        "lambda_wads",
        "lambda_rar",
        "lambda_far",
        "lambda_curvature",
        "lambda_robust",
        "lambda_attr_mass",
    ]
    negative_lambdas = [name for name in lambda_names if getattr(args, name) < 0.0]
    if negative_lambdas:
        p.error(f"Regularizer weights cannot be negative: {', '.join(negative_lambdas)}")
    baseline_mode_list = [
        mode.strip().lower()
        for mode in args.baseline_modes.split(",")
        if mode.strip()
    ]
    active_baseline_modes = set(baseline_mode_list)
    allowed_baseline_modes = {"zero", "blur", "noise", "uniform", "midpoint", "mean", "bank"}
    unknown_baselines = sorted(active_baseline_modes - allowed_baseline_modes)
    if unknown_baselines:
        p.error(f"Unknown baseline modes: {', '.join(unknown_baselines)}")
    canonical_baselines = {
        "midpoint" if mode == "mean" else mode for mode in active_baseline_modes
    }
    if len(canonical_baselines) != len(baseline_mode_list):
        p.error("--baseline_modes contains duplicate or aliased duplicate values.")
    if len(canonical_baselines) < 2 and (
        args.mode in {"main", "lambda", "baseline"}
        or args.lambda_wads > 0.0
        or args.lambda_attr_mass > 0.0
    ):
        p.error(
            "Reference-risk evaluation and regularization require at least "
            "two distinct baselines."
        )
    if "bank" in active_baseline_modes and not args.reference_bank:
        p.error("baseline mode 'bank' requires --reference_bank.")
    active_stability_modes = [
        mode.strip().lower()
        for mode in args.stability_modes.split(",")
        if mode.strip()
    ]
    allowed_stability_modes = {"noise", "brightness", "contrast", "blur", "shift"}
    unknown_stability = sorted(set(active_stability_modes) - allowed_stability_modes)
    if unknown_stability:
        p.error(f"Unknown stability modes: {', '.join(unknown_stability)}")
    if len(set(active_stability_modes)) != len(active_stability_modes):
        p.error("--stability_modes contains duplicate values.")
    if args.mode in {"main", "lambda"} and not active_stability_modes:
        p.error("--stability_modes must contain at least one perturbation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    if args.fast:
        args.epochs = min(args.epochs, 3) if args.epochs is not None else 3
        args.warmup_epochs = min(args.warmup_epochs, max(args.epochs - 1, 0))
        args.eval_batches_adv = min(args.eval_batches_adv, 3)
        args.K_attr = min(args.K_attr, 50)
        args.K_faith = min(args.K_faith, 20)
        args.ig_steps = min(args.ig_steps, 25)

    try:
        seeds = parse_seeds(args.seed, args.seeds)
        experiment_pairs = get_experiment_grid(args.dataset, args.model, args.grid)
    except ValueError as exc:
        p.error(str(exc))
    if args.eval_only and not args.load_ckpt:
        p.error("--eval_only requires --load_ckpt.")
    if args.checkpoint_path and not args.save_ckpt:
        p.error("--checkpoint_path requires --save_ckpt.")
    load_pair_count = (
        len(experiment_pairs)
        if args.mode in {"main", "faithfulness", "cert_sweep"}
        else 1
    )
    if args.load_ckpt and load_pair_count != 1:
        p.error("--load_ckpt can only be used with a single --dataset/--model setting.")
    if args.checkpoint_path and args.eval_only:
        p.error("--checkpoint_path cannot be used with --eval_only.")
    if args.checkpoint_path:
        if args.mode in {"main", "faithfulness", "cert_sweep"}:
            checkpoint_run_count = len(seeds) * len(experiment_pairs)
        elif args.mode == "lambda":
            checkpoint_run_count = len(seeds) * len(lambda_values)
        else:
            checkpoint_run_count = len(seeds)
        if checkpoint_run_count != 1:
            p.error(
                "--checkpoint_path is an exact artifact path and can only be "
                "used when the selected mode expands to one training run."
            )

    if args.mode == "main":
        out_csv = os.path.join(args.out, "table1_main.csv")
        header = [
            *CSV_IDENTITY_FIELDS,
            "dataset",
            "model",
            "seed",
            "input_profile",
            "lambda_entropy",
            *REGULARIZER_RESULT_COLUMNS,
            "baseline_modes",
            "attr_robust_baseline",
            "cert_eps",
            "train_seconds",
            "eval_seconds",
            "total_seconds",
            "clean_acc",
            "adv_error",
            "attack_suite",
            "attack_eval_n",
            "empirical_probe_rate",
            "empirical_probe_violation_rate",
            "empirical_probe_min_margin_mean",
            "crown_rate",
            "crown_proven_rate",
            "crown_conditional_rate",
            "crown_attempted_n",
            "crown_certified_n",
            "entropy_mean",
            "entropy_raw_mean",
            "ads_mean",
            "ads_raw_l2_mean",
            "ads_raw_rms_mean",
            "ads_orthogonal_rms_mean",
            "ads_output_gap_mean",
            "wads_mean",
            "raw_wads_l2_mean",
            "raw_wads_rms_mean",
            "orthogonal_wads_rms_mean",
            "baseline_output_gap_mean",
            "baseline_attr_mass_min_mean",
            "baseline_attr_mass_q05",
            "baseline_attr_mass_near_zero_rate",
            "baseline_attr_mass_ratio_min_mean",
            "baseline_attr_mass_ratio_q05",
            "baseline_attr_mass_ratio_below_floor_rate",
            "heldout_wads_mean",
            "heldout_orthogonal_wads_rms_mean",
            "heldout_attr_mass_min_mean",
            "heldout_attr_mass_ratio_min_mean",
            "heldout_attr_mass_ratio_q05",
            "heldout_attr_mass_ratio_below_floor_rate",
            "heldout_reference_valid_n",
            "ig_completeness_error_mean",
            "ig_completeness_relative_error_mean",
            "ig_completeness_abs_max_mean",
            "ig_completeness_relative_max_mean",
            "pp_stability_l2_mean",
            "pp_stability_rms_mean",
            "pp_stability_cosine_mean",
            "pp_stability_topk_jaccard_mean",
            "pp_stability_keep_rate",
            "entropy_valid_n",
            "ads_valid_n",
            "wads_valid_n",
            "pp_stability_valid_n",
            "crown_valid_n",
            "crown_error_n",
            "crown_error_types",
            "faithfulness_enabled",
            "faithfulness_baseline",
            *FAITHFULNESS_METRIC_COLUMNS,
        ]
        validate_result_header(
            header, [*REGULARIZER_RESULT_COLUMNS, *MAIN_METRIC_COLUMNS]
        )
        for seed in seeds:
            for ds, model_name in experiment_pairs:
                config_hash, _ = prepare_experiment_identity(
                    args,
                    mode="main",
                    dataset=ds,
                    model=model_name,
                    seed=seed,
                    lambda_entropy=args.lambda_entropy,
                )
                completed = csv_contains_identity(
                    out_csv,
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "config_hash": config_hash,
                    },
                )
                artifacts_complete = checkpoint_artifacts_complete(
                    args,
                    mode="main",
                    dataset=ds,
                    model_name=model_name,
                    seed=seed,
                    lam=args.lambda_entropy,
                    config_hash=config_hash,
                )
                if completed and artifacts_complete:
                    print(f"Skipping completed run: {config_hash}")
                    continue
                if completed:
                    print(
                        "Checkpoint artifact missing for completed run; "
                        f"rebuilding: {config_hash}"
                    )
                row = run_one_setting(
                    args, seed, ds, model_name, args.lambda_entropy, device, mode="main")
                if not completed:
                    append_csv(out_csv, row, header, key_fields=CSV_IDENTITY_FIELDS)
                    print("Wrote row:", row)
                else:
                    print(f"Restored checkpoint artifacts: {config_hash}")

    elif args.mode == "lambda":
        out_csv = os.path.join(args.out, "table4_lambda_ablation.csv")
        header = [
            *CSV_IDENTITY_FIELDS,
            "dataset",
            "model",
            "seed",
            "input_profile",
            "lambda_entropy",
            *REGULARIZER_RESULT_COLUMNS,
            "baseline_modes",
            "attr_robust_baseline",
            "cert_eps",
            "train_seconds",
            "eval_seconds",
            "total_seconds",
            "clean_acc",
            "adv_error",
            "attack_suite",
            "attack_eval_n",
            "empirical_probe_rate",
            "empirical_probe_violation_rate",
            "empirical_probe_min_margin_mean",
            "crown_rate",
            "crown_proven_rate",
            "crown_conditional_rate",
            "crown_attempted_n",
            "crown_certified_n",
            "entropy_mean",
            "entropy_raw_mean",
            "ads_mean",
            "ads_raw_l2_mean",
            "ads_raw_rms_mean",
            "ads_orthogonal_rms_mean",
            "ads_output_gap_mean",
            "wads_mean",
            "raw_wads_l2_mean",
            "raw_wads_rms_mean",
            "orthogonal_wads_rms_mean",
            "baseline_output_gap_mean",
            "baseline_attr_mass_min_mean",
            "baseline_attr_mass_q05",
            "baseline_attr_mass_near_zero_rate",
            "baseline_attr_mass_ratio_min_mean",
            "baseline_attr_mass_ratio_q05",
            "baseline_attr_mass_ratio_below_floor_rate",
            "heldout_wads_mean",
            "heldout_orthogonal_wads_rms_mean",
            "heldout_attr_mass_min_mean",
            "heldout_attr_mass_ratio_min_mean",
            "heldout_attr_mass_ratio_q05",
            "heldout_attr_mass_ratio_below_floor_rate",
            "heldout_reference_valid_n",
            "ig_completeness_error_mean",
            "ig_completeness_relative_error_mean",
            "ig_completeness_abs_max_mean",
            "ig_completeness_relative_max_mean",
            "pp_stability_l2_mean",
            "pp_stability_rms_mean",
            "pp_stability_cosine_mean",
            "pp_stability_topk_jaccard_mean",
            "pp_stability_keep_rate",
            "entropy_valid_n",
            "ads_valid_n",
            "wads_valid_n",
            "pp_stability_valid_n",
            "crown_valid_n",
            "crown_error_n",
            "crown_error_types",
            "faithfulness_enabled",
            "faithfulness_baseline",
            *FAITHFULNESS_METRIC_COLUMNS,
        ]
        validate_result_header(
            header, [*REGULARIZER_RESULT_COLUMNS, *MAIN_METRIC_COLUMNS]
        )
        active_dataset = args.dataset or "mnist"
        active_model = args.model or "simplecnn"
        for seed in seeds:
            for lam in lambda_values:
                config_hash, _ = prepare_experiment_identity(
                    args,
                    mode="lambda",
                    dataset=active_dataset,
                    model=active_model,
                    seed=seed,
                    lambda_entropy=lam,
                )
                completed = csv_contains_identity(
                    out_csv,
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "config_hash": config_hash,
                    },
                )
                artifacts_complete = checkpoint_artifacts_complete(
                    args,
                    mode="lambda",
                    dataset=active_dataset,
                    model_name=active_model,
                    seed=seed,
                    lam=lam,
                    config_hash=config_hash,
                )
                if completed and artifacts_complete:
                    print(f"Skipping completed run: {config_hash}")
                    continue
                if completed:
                    print(
                        "Checkpoint artifact missing for completed run; "
                        f"rebuilding: {config_hash}"
                    )
                row = run_one_setting(
                    args, seed, active_dataset, active_model, lam, device, mode="lambda")
                if not completed:
                    append_csv(out_csv, row, header, key_fields=CSV_IDENTITY_FIELDS)
                    print("Wrote row:", row)
                else:
                    print(f"Restored checkpoint artifacts: {config_hash}")

    elif args.mode == "baseline":
        out_csv = os.path.join(args.out, "table2_baseline_sensitivity.csv")
        header = [
            *CSV_IDENTITY_FIELDS,
            "dataset",
            "model",
            "seed",
            "input_profile",
            "K",
            *REGULARIZER_RESULT_COLUMNS,
            "ads_zero_blur",
            "ads_zero_noise",
            "ads_zero_uniform",
            "ads_blur_noise",
            "ads_blur_uniform",
            "ads_noise_uniform",
        ]
        validate_result_header(header, REGULARIZER_RESULT_COLUMNS)
        active_dataset = args.dataset or "mnist"
        active_model = args.model or "simplecnn"
        for seed in seeds:
            config_hash, _ = prepare_experiment_identity(
                args,
                mode="baseline",
                dataset=active_dataset,
                model=active_model,
                seed=seed,
                lambda_entropy=args.lambda_entropy,
            )
            completed = csv_contains_identity(
                out_csv,
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "config_hash": config_hash,
                },
            )
            artifacts_complete = checkpoint_artifacts_complete(
                args,
                mode="baseline",
                dataset=active_dataset,
                model_name=active_model,
                seed=seed,
                lam=args.lambda_entropy,
                config_hash=config_hash,
            )
            if completed and artifacts_complete:
                print(f"Skipping completed run: {config_hash}")
                continue
            if completed:
                print(
                    "Checkpoint artifact missing for completed run; "
                    f"rebuilding: {config_hash}"
                )
            row = run_one_setting(args, seed, active_dataset, active_model,
                                  args.lambda_entropy, device, mode="baseline")
            if not completed:
                append_csv(out_csv, row, header, key_fields=CSV_IDENTITY_FIELDS)
                print("Wrote row:", row)
            else:
                print(f"Restored checkpoint artifacts: {config_hash}")

    elif args.mode == "faithfulness":
        out_csv = os.path.join(args.out, "tables5_6_faithfulness.csv")
        header = [
            *CSV_IDENTITY_FIELDS,
            "dataset",
            "model",
            "seed",
            "input_profile",
            "K",
            "faithfulness_baseline",
            *REGULARIZER_RESULT_COLUMNS,
            "ig_del_auc_mean",
            "ig_ins_auc_mean",
            "sg2_del_auc_mean",
            "sg2_ins_auc_mean",
            "random_del_auc_mean",
            "random_ins_auc_mean",
            "ig_valid_n",
            "sg2_valid_n",
            "random_valid_n",
        ]
        validate_result_header(header, REGULARIZER_RESULT_COLUMNS)
        for seed in seeds:
            for ds, model_name in experiment_pairs:
                config_hash, _ = prepare_experiment_identity(
                    args,
                    mode="faithfulness",
                    dataset=ds,
                    model=model_name,
                    seed=seed,
                    lambda_entropy=args.lambda_entropy,
                )
                completed = csv_contains_identity(
                    out_csv,
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "config_hash": config_hash,
                    },
                )
                artifacts_complete = checkpoint_artifacts_complete(
                    args,
                    mode="faithfulness",
                    dataset=ds,
                    model_name=model_name,
                    seed=seed,
                    lam=args.lambda_entropy,
                    config_hash=config_hash,
                )
                if completed and artifacts_complete:
                    print(f"Skipping completed run: {config_hash}")
                    continue
                if completed:
                    print(
                        "Checkpoint artifact missing for completed run; "
                        f"rebuilding: {config_hash}"
                    )
                row = run_one_setting(
                    args, seed, ds, model_name, args.lambda_entropy, device, mode="faithfulness")
                if not completed:
                    append_csv(out_csv, row, header, key_fields=CSV_IDENTITY_FIELDS)
                    print("Wrote row:", row)
                else:
                    print(f"Restored checkpoint artifacts: {config_hash}")

    elif args.mode == "cert_sweep":
        out_csv = os.path.join(args.out, "table_certification_sweep.csv")
        header = [
            *CSV_IDENTITY_FIELDS,
            "dataset",
            "model",
            "seed",
            "input_profile",
            "lambda_entropy",
            *REGULARIZER_RESULT_COLUMNS,
            "cert_pixel_eps",
            "cert_eps",
            "crown_rate",
            "crown_proven_rate",
            "crown_conditional_rate",
            "cert_attempted_n",
            "cert_certified_n",
            "cert_valid_n",
            "cert_error_n",
            "cert_error_types",
        ]
        validate_result_header(header, REGULARIZER_RESULT_COLUMNS)
        for seed in seeds:
            for ds, model_name in experiment_pairs:
                config_hash, _ = prepare_experiment_identity(
                    args,
                    mode="cert_sweep",
                    dataset=ds,
                    model=model_name,
                    seed=seed,
                    lambda_entropy=args.lambda_entropy,
                )
                pending_pixel_values = [
                    pixel_value
                    for pixel_value in cert_pixel_values
                    if not csv_contains_identity(
                        out_csv,
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "config_hash": config_hash,
                            "cert_pixel_eps": pixel_value,
                        },
                    )
                ]
                artifacts_complete = checkpoint_artifacts_complete(
                    args,
                    mode="cert_sweep",
                    dataset=ds,
                    model_name=model_name,
                    seed=seed,
                    lam=args.lambda_entropy,
                    config_hash=config_hash,
                )
                if not pending_pixel_values:
                    if artifacts_complete:
                        print(f"Skipping completed certification sweep: {config_hash}")
                        continue
                    print(
                        "Checkpoint artifact missing for completed certification "
                        f"sweep; rebuilding: {config_hash}"
                    )
                    pending_pixel_values = cert_pixel_values
                rows = run_one_setting(
                    args,
                    seed,
                    ds,
                    model_name,
                    args.lambda_entropy,
                    device,
                    mode="cert_sweep",
                    cert_pixel_values=pending_pixel_values,
                )
                for row in rows:
                    append_csv(
                        out_csv,
                        row,
                        header,
                        key_fields=[*CSV_IDENTITY_FIELDS, "cert_pixel_eps"],
                        duplicate_policy="skip",
                    )
                    print("Wrote row:", row)


if __name__ == "__main__":
    main()
