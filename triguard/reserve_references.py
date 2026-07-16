"""Create a model-independent candidate pool for reference selection."""

import argparse
import os

import torch

from .data import get_dataset
from .models import uses_imagenet_preprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidate_count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_root", default="./data")
    args = parser.parse_args()

    input_profile = (
        "imagenet" if uses_imagenet_preprocessing(args.model) else "native"
    )
    train_set, _, _, _, _, _ = get_dataset(
        args.dataset,
        data_root=args.data_root,
        input_profile=input_profile,
    )
    if not 4 <= args.candidate_count < len(train_set):
        parser.error("--candidate_count must be at least 4 and smaller than the train split.")
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(train_set), generator=generator)[
        : args.candidate_count
    ]
    payload = {
        "indices": indices,
        "metadata": {
            "dataset": args.dataset.lower(),
            "input_profile": input_profile,
            "source_split": "train",
            "selection": "model_independent_random_reservation",
            "seed": args.seed,
        },
    }
    directory = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(directory, exist_ok=True)
    temporary = f"{args.out}.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, args.out)
    print(f"Wrote reserved reference-candidate indices: {args.out}")


if __name__ == "__main__":
    main()
