from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    build_dataloaders,
    evaluate_model,
    load_session_splits,
    num_classes_and_blank,
    resolve_project_path,
)
from models import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--split-config", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    train_config = checkpoint.get("config", {})
    if not train_config:
        raise RuntimeError("Checkpoint does not contain training config.")

    model_config = argparse.Namespace(**train_config)
    data_dir = args.data_dir or train_config.get("data_dir", "data")
    split_config = args.split_config or train_config.get(
        "split_config",
        "emg2qwerty/config/user/single_user.yaml",
    )
    batch_size = args.batch_size or int(train_config.get("batch_size", 8))
    window_length = int(train_config.get("window_length", 8000))
    left_padding = int(train_config.get("left_padding", 0))
    right_padding = int(train_config.get("right_padding", 0))
    num_channels = int(train_config.get("num_channels", 32))
    downsample_factor = int(train_config.get("downsample_factor", 1))
    train_fraction = float(train_config.get("train_fraction", 1.0))
    seed = int(train_config.get("seed", 42))

    sessions = load_session_splits(
        split_config_path=split_config,
        data_dir=data_dir,
        train_fraction=train_fraction,
        seed=seed,
    )
    dataloaders = build_dataloaders(
        sessions=sessions,
        batch_size=batch_size,
        num_workers=args.num_workers,
        window_length=window_length,
        left_padding=left_padding,
        right_padding=right_padding,
        num_channels=num_channels,
        downsample_factor=downsample_factor,
        augment=False,
        temporal_jitter=0,
    )

    num_classes, blank_index = num_classes_and_blank()
    num_classes = int(checkpoint.get("num_classes", num_classes))
    blank_index = int(checkpoint.get("blank_index", blank_index))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_config, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = nn.CTCLoss(blank=blank_index, reduction="mean", zero_infinity=True)

    metrics = evaluate_model(
        model=model,
        loader=dataloaders[args.split],
        criterion=criterion,
        device=device,
        blank_index=blank_index,
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {args.split}")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"CER: {metrics['cer']:.2f}")


if __name__ == "__main__":
    main()
