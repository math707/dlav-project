"""Phase 2 submission helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _extract_trajectory_prediction(outputs):
    if isinstance(outputs, dict):
        trajectory = outputs.get('trajectory')
        if trajectory is None:
            raise RuntimeError("Model output dict did not contain a 'trajectory' key.")
        return trajectory
    if isinstance(outputs, tuple):
        if not outputs:
            raise RuntimeError('Model output tuple was empty.')
        return outputs[0]
    return outputs


def predict_future_plans(model, data_loader, device) -> np.ndarray:
    model = model.to(device)
    model.eval()

    all_plans = []
    with torch.no_grad():
        for batch in data_loader:
            camera = batch['camera'].to(device)
            history = batch['history'].to(device)
            driving_command = batch['driving_command'].to(device)

            outputs = model(camera, history, driving_command)
            trajectory = _extract_trajectory_prediction(outputs)
            all_plans.append(trajectory[..., :2].cpu().numpy())

    return np.concatenate(all_plans, axis=0)


def build_submission_dataframe(all_plans: np.ndarray) -> pd.DataFrame:
    total_samples, timesteps, dimensions = all_plans.shape
    if dimensions != 2:
        raise ValueError(f'Phase 2 submissions expect XY trajectories with 2 dimensions, got {dimensions}.')

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
    data_loader,
    device,
    output_path: str | Path,
    legacy_output_path: str | Path | None = None,
    copy_fn=None,
) -> pd.DataFrame:
    submission = build_submission_dataframe(predict_future_plans(model, data_loader, device))

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
