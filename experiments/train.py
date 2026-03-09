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
    parse_int_list,
    resolve_project_path,
    set_seed,
)
from models import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EMG2QWERTY model with a minimal CLI."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing subject HDF5 files.",
    )
    parser.add_argument(
        "--split-config",
        type=str,
        default="emg2qwerty/config/user/single_user.yaml",
        help="Path to train/val/test split YAML.",
    )
    parser.add_argument("--model-type", type=str, default="rnn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="reduce_on_plateau",
        choices=["none", "reduce_on_plateau", "cosine"],
    )
    parser.add_argument("--window-length", type=int, default=8000)
    parser.add_argument("--left-padding", type=int, default=0)
    parser.add_argument("--right-padding", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--num-channels", type=int, default=32)
    parser.add_argument("--downsample-factor", type=int, default=1)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--temporal-jitter", type=int, default=120)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    # RNN args
    parser.add_argument("--cell-type", type=str, default="lstm")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--unidirectional", action="store_true")
    parser.add_argument("--input-proj-dim", type=int, default=64)

    # CNN+RNN args
    parser.add_argument("--conv-channels", type=str, default="64,128")
    parser.add_argument("--conv-kernel-size", type=int, default=7)
    parser.add_argument("--conv-stride", type=int, default=2)
    parser.add_argument("--conv-dropout", type=float, default=0.1)
    parser.add_argument("--num-rnn-layers", type=int, default=2)
    parser.add_argument("--rnn-dropout", type=float, default=0.2)

    # Transformer args
    parser.add_argument("--frontend-channels", type=int, default=64)
    parser.add_argument("--frontend-kernel-size", type=int, default=7)
    parser.add_argument("--frontend-stride", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    return parser.parse_args()


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        inputs = batch["inputs"].to(device).float()
        targets = batch["targets"].to(device).long()
        input_lengths = batch["input_lengths"].to(device).long()
        target_lengths = batch["target_lengths"].to(device).long()

        optimizer.zero_grad(set_to_none=True)
        log_probs = model(inputs)
        emission_lengths = model.output_lengths(input_lengths).to(device).long()
        if torch.any(emission_lengths < target_lengths):
            raise RuntimeError("Found emission_lengths < target_lengths")

        loss = criterion(
            log_probs,
            targets.transpose(0, 1),
            emission_lengths,
            target_lengths,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        running_loss += float(loss.item())
        num_batches += 1

    return running_loss / max(num_batches, 1)


def main() -> None:
    args = parse_args()
    args.bidirectional = not args.unidirectional
    args.conv_channels = parse_int_list(args.conv_channels)
    args.data_dir = str(resolve_project_path(args.data_dir))
    args.split_config = str(resolve_project_path(args.split_config))
    args.checkpoint_dir = str(resolve_project_path(args.checkpoint_dir))

    if args.num_channels < 1 or args.num_channels > 32:
        raise ValueError("--num-channels must be in [1, 32]")
    if args.downsample_factor < 1:
        raise ValueError("--downsample-factor must be >= 1")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sessions = load_session_splits(
        split_config_path=args.split_config,
        data_dir=args.data_dir,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    dataloaders = build_dataloaders(
        sessions=sessions,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        window_length=args.window_length,
        left_padding=args.left_padding,
        right_padding=args.right_padding,
        num_channels=args.num_channels,
        downsample_factor=args.downsample_factor,
        augment=args.augment,
        temporal_jitter=args.temporal_jitter,
    )

    num_classes, blank_index = num_classes_and_blank()
    model = get_model(args, num_classes=num_classes).to(device)
    criterion = nn.CTCLoss(blank=blank_index, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if args.scheduler == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"{args.model_type}_best.pt"

    best_val_cer = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_metrics = evaluate_model(
            model=model,
            loader=dataloaders["val"],
            criterion=criterion,
            device=device,
            blank_index=blank_index,
        )

        if scheduler is not None:
            if args.scheduler == "reduce_on_plateau":
                scheduler.step(val_metrics["cer"])
            else:
                scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_cer={val_metrics['cer']:.2f} | "
            f"lr={lr:.2e}"
        )

        if val_metrics["cer"] < best_val_cer:
            best_val_cer = val_metrics["cer"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": args.model_type,
                    "num_classes": num_classes,
                    "blank_index": blank_index,
                    "best_val_cer": best_val_cer,
                    "epoch": epoch,
                    "config": vars(args),
                },
                best_path,
            )
            print(f"Saved new best checkpoint to {best_path}")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate_model(
        model=model,
        loader=dataloaders["test"],
        criterion=criterion,
        device=device,
        blank_index=blank_index,
    )
    print(f"Best checkpoint: {best_path}")
    print(f"Best val CER: {best_val_cer:.2f}")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Test CER: {test_metrics['cer']:.2f}")


if __name__ == "__main__":
    main()
