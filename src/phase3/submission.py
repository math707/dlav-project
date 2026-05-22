"""Phase 3 submission helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import DrivingDataset, list_pkl_files


def _resolve_device(device=None):
    return device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _resolve_public_test_loader(
    data_loader,
    *,
    public_test_dir: str | Path | None,
    test_dir: str | Path | None,
    test_batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    if data_loader is not None:
        return data_loader

    resolved_public_test_dir = public_test_dir if public_test_dir is not None else test_dir
    if resolved_public_test_dir is None:
        raise ValueError('Provide either data_loader or public_test_dir.')
    return build_public_test_data_loader(
        resolved_public_test_dir,
        batch_size=test_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def list_test_public_real_files(test_dir: str | Path) -> list[Path]:
    """Load public-test files in sorted numeric order for Kaggle submission generation."""

    test_files = list_pkl_files(test_dir)
    if not test_files:
        raise FileNotFoundError(f'No .pkl files found in {Path(test_dir)}')
    return test_files


def build_public_test_data_loader(
    public_test_dir: str | Path,
    *,
    batch_size: int = 250,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    test_dataset = DrivingDataset(list_test_public_real_files(public_test_dir), test=True)
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_test_data_loader(
    test_dir: str | Path,
    *,
    batch_size: int = 250,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """Backward-compatible alias for the Phase 3 public test loader builder."""

    return build_public_test_data_loader(
        test_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def predict_future_plans(model, data_loader, device) -> np.ndarray:
    device = _resolve_device(device)
    model = model.to(device)
    model.eval()

    all_plans = []
    with torch.no_grad():
        for batch in data_loader:
            camera = batch['camera'].to(device)
            history = batch['history'].to(device)

            pred_future = model(camera, history)
            all_plans.append(pred_future[..., :2].cpu().numpy())

    return np.concatenate(all_plans, axis=0)


def load_models_from_checkpoints(
    checkpoint_paths: Sequence[str | Path],
    *,
    model_factory: Callable[[], torch.nn.Module],
    device=None,
):
    device = _resolve_device(device)
    models = []

    for checkpoint_path in checkpoint_paths:
        resolved_checkpoint_path = Path(checkpoint_path)
        if not resolved_checkpoint_path.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {resolved_checkpoint_path}')

        model = model_factory()
        state_dict = torch.load(resolved_checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        models.append(model)

    if not models:
        raise ValueError('At least one checkpoint is required.')

    return models


def predict_future_plans_ensemble(models, data_loader, device) -> np.ndarray:
    device = _resolve_device(device)
    all_plans = []

    for model in models:
        model.to(device)
        model.eval()

    with torch.no_grad():
        for batch in data_loader:
            camera = batch['camera'].to(device)
            history = batch['history'].to(device)
            predictions = [model(camera, history)[..., :2] for model in models]
            ensemble_prediction = torch.stack(predictions, dim=0).mean(dim=0)
            all_plans.append(ensemble_prediction.cpu().numpy())

    if not all_plans:
        raise ValueError('Data loader produced no samples.')

    return np.concatenate(all_plans, axis=0)


def build_submission_dataframe(all_plans: np.ndarray) -> pd.DataFrame:
    total_samples, timesteps, dimensions = all_plans.shape
    if timesteps != 60 or dimensions != 2:
        raise ValueError(
            f'Phase 3 submissions expect trajectories with shape [N, 60, 2], got {all_plans.shape}.'
        )

    flattened_plans = all_plans.reshape(total_samples, timesteps * dimensions)
    submission = pd.DataFrame(flattened_plans)
    submission.insert(0, 'id', np.arange(total_samples))

    column_names = ['id']
    for timestep in range(1, timesteps + 1):
        column_names.append(f'x_{timestep}')
        column_names.append(f'y_{timestep}')
    submission.columns = column_names
    return submission


def generate_submission(
    model,
    *,
    output_path: str | Path,
    device,
    data_loader=None,
    public_test_dir: str | Path | None = None,
    test_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    reload_checkpoint: bool = False,
    legacy_output_path: str | Path | None = None,
    copy_fn=None,
    expected_num_samples: int | None = None,
    test_batch_size: int = 250,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> pd.DataFrame:
    device = _resolve_device(device)

    if reload_checkpoint:
        if checkpoint_path is None:
            raise ValueError('reload_checkpoint=True requires checkpoint_path.')
        state_dict = torch.load(Path(checkpoint_path), map_location=device)
        model.load_state_dict(state_dict)

    data_loader = _resolve_public_test_loader(
        data_loader,
        public_test_dir=public_test_dir,
        test_dir=test_dir,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    submission = build_submission_dataframe(predict_future_plans(model, data_loader, device))

    if expected_num_samples is not None:
        expected_shape = (expected_num_samples, 121)
        if submission.shape != expected_shape:
            raise ValueError(f'Expected submission shape {expected_shape}, got {submission.shape}.')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    if legacy_output_path is not None:
        legacy_output_path = Path(legacy_output_path)
        if copy_fn is not None:
            copy_fn(output_path, legacy_output_path)
        else:
            legacy_output_path.parent.mkdir(parents=True, exist_ok=True)
            submission.to_csv(legacy_output_path, index=False)

    return submission


def generate_ensemble_submission(
    models,
    *,
    output_path: str | Path,
    device,
    data_loader=None,
    public_test_dir: str | Path | None = None,
    test_dir: str | Path | None = None,
    legacy_output_path: str | Path | None = None,
    copy_fn=None,
    expected_num_samples: int | None = None,
    test_batch_size: int = 250,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> pd.DataFrame:
    device = _resolve_device(device)
    data_loader = _resolve_public_test_loader(
        data_loader,
        public_test_dir=public_test_dir,
        test_dir=test_dir,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    submission = build_submission_dataframe(predict_future_plans_ensemble(models, data_loader, device))

    if expected_num_samples is not None:
        expected_shape = (expected_num_samples, 121)
        if submission.shape != expected_shape:
            raise ValueError(f'Expected submission shape {expected_shape}, got {submission.shape}.')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    if legacy_output_path is not None:
        legacy_output_path = Path(legacy_output_path)
        if copy_fn is not None:
            copy_fn(output_path, legacy_output_path)
        else:
            legacy_output_path.parent.mkdir(parents=True, exist_ok=True)
            submission.to_csv(legacy_output_path, index=False)

    return submission
