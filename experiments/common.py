from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import editdistance
import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "emg2qwerty"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.transforms import (
    Compose,
    ForEach,
    RandomBandRotation,
    TemporalAlignmentJitter,
    ToTensor,
)


@dataclass
class PostProcessEMG:
    num_channels: int = 32
    downsample_factor: int = 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor
        if self.downsample_factor > 1:
            x = x[:: self.downsample_factor]

        if self.num_channels < 32:
            x = x.clone()
            t_steps = x.shape[0]
            flattened = x.reshape(t_steps, 32)
            flattened[:, self.num_channels :] = 0
            x = flattened.reshape(t_steps, 2, 16)

        return x.float()


def parse_int_list(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return [int(v) for v in value]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_session_splits(
    split_config_path: str | Path,
    data_dir: str | Path,
    train_fraction: float = 1.0,
    seed: int = 42,
) -> dict[str, list[Path]]:
    split_config_path = resolve_project_path(split_config_path)
    data_dir = resolve_project_path(data_dir)

    with split_config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_cfg = config["dataset"]
    split_sessions: dict[str, list[dict[str, str]]] = {
        "train": list(dataset_cfg.get("train", [])),
        "val": list(dataset_cfg.get("val", [])),
        "test": list(dataset_cfg.get("test", [])),
    }

    if train_fraction < 1.0:
        original = split_sessions["train"]
        sample_size = max(1, int(round(len(original) * train_fraction)))
        rng = random.Random(seed)
        split_sessions["train"] = rng.sample(original, sample_size)

    resolved: dict[str, list[Path]] = {}
    missing: list[Path] = []
    for split, entries in split_sessions.items():
        paths: list[Path] = []
        for entry in entries:
            session = entry["session"]
            path = data_dir / f"{session}.hdf5"
            if not path.exists():
                missing.append(path)
            paths.append(path)
        resolved[split] = paths

    if missing:
        example = "\n".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            "Missing session files under data_dir. "
            f"Example missing files:\n{example}"
        )

    return resolved


def build_transform(
    *,
    is_train: bool,
    num_channels: int,
    downsample_factor: int,
    augment: bool,
    temporal_jitter: int,
) -> Compose:
    transforms = [ToTensor()]
    if is_train and augment:
        transforms.append(ForEach(RandomBandRotation(offsets=(-1, 0, 1))))
        if temporal_jitter > 0:
            transforms.append(TemporalAlignmentJitter(max_offset=temporal_jitter))
    transforms.append(
        PostProcessEMG(
            num_channels=num_channels,
            downsample_factor=downsample_factor,
        )
    )
    return Compose(transforms)


def _make_concat_dataset(datasets: list[WindowedEMGDataset]):
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def build_dataloaders(
    *,
    sessions: dict[str, list[Path]],
    batch_size: int,
    num_workers: int,
    window_length: int,
    left_padding: int,
    right_padding: int,
    num_channels: int,
    downsample_factor: int,
    augment: bool,
    temporal_jitter: int,
) -> dict[str, DataLoader]:
    train_transform = build_transform(
        is_train=True,
        num_channels=num_channels,
        downsample_factor=downsample_factor,
        augment=augment,
        temporal_jitter=temporal_jitter,
    )
    eval_transform = build_transform(
        is_train=False,
        num_channels=num_channels,
        downsample_factor=downsample_factor,
        augment=False,
        temporal_jitter=0,
    )

    train_dataset = _make_concat_dataset(
        [
            WindowedEMGDataset(
                hdf5_path=path,
                transform=train_transform,
                window_length=window_length,
                padding=(left_padding, right_padding),
                jitter=True,
            )
            for path in sessions["train"]
        ]
    )

    val_dataset = _make_concat_dataset(
        [
            WindowedEMGDataset(
                hdf5_path=path,
                transform=eval_transform,
                window_length=window_length,
                padding=(left_padding, right_padding),
                jitter=False,
            )
            for path in sessions["val"]
        ]
    )

    test_dataset = _make_concat_dataset(
        [
            WindowedEMGDataset(
                hdf5_path=path,
                transform=eval_transform,
                window_length=None,
                padding=(0, 0),
                jitter=False,
            )
            for path in sessions["test"]
        ]
    )

    persistent_workers = num_workers > 0
    pin_memory = torch.cuda.is_available()

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        ),
    }


def greedy_decode(
    log_probs: torch.Tensor,
    emission_lengths: torch.Tensor,
    blank_index: int,
) -> list[LabelData]:
    labels = log_probs.argmax(dim=-1).detach().cpu().numpy()
    emission_lengths = emission_lengths.detach().cpu().numpy()

    outputs: list[LabelData] = []
    batch_size = labels.shape[1]
    for batch_idx in range(batch_size):
        decoded: list[int] = []
        prev = blank_index
        t_max = int(emission_lengths[batch_idx])
        for t in range(t_max):
            current = int(labels[t, batch_idx])
            if current != blank_index and current != prev:
                decoded.append(current)
            prev = current
        outputs.append(LabelData.from_labels(decoded))
    return outputs


def compute_edit_counts(
    predictions: list[LabelData],
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
) -> tuple[int, int]:
    targets_np = targets.detach().cpu().numpy()
    target_lengths_np = target_lengths.detach().cpu().numpy()

    edits = 0
    total_chars = 0
    for i, pred in enumerate(predictions):
        reference = LabelData.from_labels(targets_np[: target_lengths_np[i], i].tolist())
        edits += editdistance.eval(pred.text, reference.text)
        total_chars += len(reference.text)
    return edits, total_chars


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.CTCLoss,
    device: torch.device,
    blank_index: int,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    num_batches = 0
    total_edits = 0
    total_chars = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device).float()
            targets = batch["targets"].to(device).long()
            input_lengths = batch["input_lengths"].to(device).long()
            target_lengths = batch["target_lengths"].to(device).long()

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
            running_loss += float(loss.item())
            num_batches += 1

            predictions = greedy_decode(log_probs, emission_lengths, blank_index)
            edits, chars = compute_edit_counts(predictions, targets, target_lengths)
            total_edits += edits
            total_chars += chars

    avg_loss = running_loss / max(num_batches, 1)
    cer = 100.0 * total_edits / max(total_chars, 1)
    return {"loss": avg_loss, "cer": cer}


def num_classes_and_blank() -> tuple[int, int]:
    char_set = charset()
    return char_set.num_classes, char_set.null_class
