"""Phase 3 dataset helpers for sim-to-real trajectory planning."""

from __future__ import annotations

import pickle
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


PHASE3_DATASET_SPECS = {
    'train': {
        'file_id': '1YkGwaxBKNiYL2nq--cB6WMmYGzRmRKVr',
        'zip_name': 'dlav_train.zip',
        'target_subdir': 'train',
    },
    'val_real': {
        'file_id': '17DREGym_-v23f_qbkMHr7vJstbuTt0if',
        'zip_name': 'dlav_val_real.zip',
        'target_subdir': 'val_real',
    },
    'test_public_real': {
        'file_id': '1_l6cui0pCJ_caixN0uTkkUOfu6ICO8u5',
        'zip_name': 'test_public_real.zip',
        'target_subdir': 'test_public_real',
    },
}


def _numeric_sort_key(path: Path) -> tuple[int, int | str]:
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


def list_pkl_files(directory: str | Path) -> list[Path]:
    """Return all `.pkl` files in a directory using stable numeric ordering."""

    directory = Path(directory)
    return sorted(directory.glob('*.pkl'), key=_numeric_sort_key)


def split_real_train_val(
    real_dir: str | Path,
    *,
    real_train_count: int = 500,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Randomly split real labeled data into train and validation subsets."""

    real_files = list_pkl_files(real_dir)
    if not real_files:
        raise FileNotFoundError(f'No .pkl files found in {Path(real_dir)}')
    if real_train_count < 0 or real_train_count > len(real_files):
        raise ValueError(
            f'real_train_count must be between 0 and {len(real_files)}, got {real_train_count}.'
        )

    shuffled_files = list(real_files)
    random.Random(seed).shuffle(shuffled_files)
    real_train_files = sorted(shuffled_files[:real_train_count], key=_numeric_sort_key)
    real_val_files = sorted(shuffled_files[real_train_count:], key=_numeric_sort_key)
    return real_train_files, real_val_files


def build_phase3_splits(
    synthetic_train_dir: str | Path,
    real_dir: str | Path,
    *,
    real_train_count: int = 500,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Build the synthetic-train, real-train, real-val, and mixed-train file lists."""

    synthetic_train_files = list_pkl_files(synthetic_train_dir)
    if not synthetic_train_files:
        raise FileNotFoundError(f'No .pkl files found in {Path(synthetic_train_dir)}')

    real_train_files, real_val_files = split_real_train_val(
        real_dir,
        real_train_count=real_train_count,
        seed=seed,
    )

    return {
        'synthetic_train': synthetic_train_files,
        'real_train': real_train_files,
        'real_val': real_val_files,
        'mixed_train': [*synthetic_train_files, *real_train_files],
    }


def _format_camera_tensor(camera: Any) -> torch.Tensor:
    camera_tensor = torch.as_tensor(camera)
    if camera_tensor.ndim != 3:
        raise ValueError(f'Unsupported camera shape {tuple(camera_tensor.shape)}. Expected 3 dimensions.')

    if camera_tensor.shape[-1] == 3:
        camera_tensor = camera_tensor.permute(2, 0, 1)
    elif camera_tensor.shape[0] != 3:
        raise ValueError(
            f'Unsupported camera shape {tuple(camera_tensor.shape)}. Expected [H, W, 3] or [3, H, W].'
        )

    camera_tensor = camera_tensor.contiguous().to(torch.float32)
    if camera_tensor.detach().amax().item() > 1.5:
        camera_tensor = camera_tensor / 255.0
    return camera_tensor


def _format_history_tensor(history: Any) -> torch.Tensor:
    history_tensor = torch.as_tensor(history, dtype=torch.float32)
    if history_tensor.shape != (21, 3):
        raise ValueError(
            f'Unsupported history shape {tuple(history_tensor.shape)}. Expected exactly [21, 3].'
        )
    return history_tensor


def _format_future_tensor(future: Any, *, xy_only: bool) -> torch.Tensor:
    future_tensor = torch.as_tensor(future, dtype=torch.float32)
    if future_tensor.ndim != 2 or future_tensor.shape[0] != 60 or future_tensor.shape[1] < 2:
        raise ValueError(
            f'Unsupported future shape {tuple(future_tensor.shape)}. Expected [60, >=2].'
        )
    if xy_only:
        future_tensor = future_tensor[:, :2]
    return future_tensor


class DrivingDataset(Dataset):
    """Phase 3 dataset using camera + SDC history and optional future XY labels."""

    def __init__(
        self,
        file_list: Sequence[str | Path],
        *,
        test: bool = False,
        image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        future_xy_only: bool = True,
    ):
        self.samples = [Path(path) for path in file_list]
        self.test = test
        self.image_transform = image_transform
        self.future_xy_only = future_xy_only

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample_path = self.samples[idx]
        with sample_path.open('rb') as file:
            data = pickle.load(file)

        camera = _format_camera_tensor(data['camera'])
        if self.image_transform is not None:
            camera = self.image_transform(camera)

        batch = {
            'camera': camera,
            'history': _format_history_tensor(data['sdc_history_feature']),
        }

        if self.test:
            return batch

        if 'sdc_future_feature' not in data:
            raise KeyError(f"Sample {sample_path} does not contain 'sdc_future_feature'.")

        batch['future'] = _format_future_tensor(data['sdc_future_feature'], xy_only=self.future_xy_only)
        return batch
