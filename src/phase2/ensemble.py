"""Phase 2 checkpoint-ensemble helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .model import build_model
from .submission import build_submission_dataframe


@dataclass(frozen=True)
class EnsembleMemberSpec:
    checkpoint_path: str | Path
    model_name: str
    model_kwargs: dict[str, Any] | None = None


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


def _resolve_device(device=None):
    return device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _coerce_model_kwargs(model_kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(model_kwargs or {})


def _coerce_member_specs(
    members: Sequence[str | Path | EnsembleMemberSpec | Mapping[str, Any]],
    *,
    model_name: str | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> list[EnsembleMemberSpec]:
    if not members:
        raise ValueError('At least one ensemble member is required.')

    shared_model_kwargs = _coerce_model_kwargs(model_kwargs)
    specs: list[EnsembleMemberSpec] = []

    for member in members:
        if isinstance(member, EnsembleMemberSpec):
            specs.append(
                EnsembleMemberSpec(
                    checkpoint_path=member.checkpoint_path,
                    model_name=member.model_name,
                    model_kwargs=_coerce_model_kwargs(member.model_kwargs),
                )
            )
            continue

        if isinstance(member, Mapping):
            checkpoint_path = member.get('checkpoint_path')
            resolved_model_name = member.get('model_name', model_name)
            if checkpoint_path is None:
                raise ValueError("Each ensemble member mapping must include 'checkpoint_path'.")
            if resolved_model_name is None:
                raise ValueError(
                    "Each ensemble member mapping must include 'model_name' or you must pass model_name=..."
                )

            member_kwargs = shared_model_kwargs.copy()
            member_kwargs.update(_coerce_model_kwargs(member.get('model_kwargs')))
            specs.append(
                EnsembleMemberSpec(
                    checkpoint_path=checkpoint_path,
                    model_name=resolved_model_name,
                    model_kwargs=member_kwargs,
                )
            )
            continue

        if model_name is None:
            raise ValueError("When passing raw checkpoint paths, model_name must also be provided.")
        specs.append(
            EnsembleMemberSpec(
                checkpoint_path=member,
                model_name=model_name,
                model_kwargs=shared_model_kwargs.copy(),
            )
        )

    return specs


def load_ensemble_models(
    members: Sequence[str | Path | EnsembleMemberSpec | Mapping[str, Any]],
    *,
    model_name: str | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    device=None,
):
    """Build and load a list of Phase 2 models from checkpoint specs."""

    device = _resolve_device(device)
    specs = _coerce_member_specs(members, model_name=model_name, model_kwargs=model_kwargs)
    models = []

    for spec in specs:
        checkpoint_path = Path(spec.checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

        model = build_model(spec.model_name, **_coerce_model_kwargs(spec.model_kwargs))
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        models.append(model)

    return models, specs


def _predict_ensemble_batch(models, camera, history, driving_command):
    predictions = []
    for model in models:
        outputs = model(camera, history, driving_command)
        trajectory = _extract_trajectory_prediction(outputs)
        predictions.append(trajectory[..., :2])

    stacked_predictions = torch.stack(predictions, dim=0)
    return stacked_predictions.mean(dim=0)


def predict_future_plans_ensemble(models, data_loader, *, device=None) -> np.ndarray:
    """Average XY predictions from a list of already-loaded Phase 2 models."""

    device = _resolve_device(device)
    all_plans = []

    for model in models:
        model.to(device)
        model.eval()

    with torch.no_grad():
        for batch in data_loader:
            camera = batch['camera'].to(device)
            history = batch['history'].to(device)
            driving_command = batch['driving_command'].to(device)
            ensemble_prediction = _predict_ensemble_batch(models, camera, history, driving_command)
            all_plans.append(ensemble_prediction.cpu().numpy())

    if not all_plans:
        raise ValueError('Data loader produced no samples.')

    return np.concatenate(all_plans, axis=0)


def validate_ensemble(models, val_loader, *, device=None) -> dict[str, float]:
    """Compute XY-only validation metrics for an already-loaded ensemble."""

    device = _resolve_device(device)
    total_samples = 0
    total_traj_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0

    for model in models:
        model.to(device)
        model.eval()

    with torch.no_grad():
        for batch in val_loader:
            camera = batch['camera'].to(device)
            history = batch['history'].to(device)
            driving_command = batch['driving_command'].to(device)
            future_xy = batch['future'].to(device)[..., :2]

            ensemble_prediction = _predict_ensemble_batch(models, camera, history, driving_command)

            batch_size = future_xy.size(0)
            traj_loss = torch.nn.functional.mse_loss(ensemble_prediction, future_xy)
            ade = torch.norm(ensemble_prediction - future_xy, p=2, dim=-1).mean(dim=1).sum()
            fde = torch.norm(ensemble_prediction[:, -1, :] - future_xy[:, -1, :], p=2, dim=-1).sum()

            total_samples += batch_size
            total_traj_loss += traj_loss.item() * batch_size
            total_ade += ade.item()
            total_fde += fde.item()

    if total_samples == 0:
        raise ValueError('Validation loader produced no samples.')

    return {
        'val_traj_loss': total_traj_loss / total_samples,
        'val_ADE': total_ade / total_samples,
        'val_FDE': total_fde / total_samples,
    }


def generate_ensemble_submission(
    models,
    data_loader,
    *,
    device=None,
    output_path: str | Path,
    legacy_output_path: str | Path | None = None,
    copy_fn=None,
) -> pd.DataFrame:
    """Generate a submission CSV from averaged ensemble predictions."""

    submission = build_submission_dataframe(predict_future_plans_ensemble(models, data_loader, device=device))

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
