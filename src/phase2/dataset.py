"""Phase 2 dataset helpers for perception-aware planning."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


DRIVING_COMMAND_TO_ID = {
    'forward': 0,
    'straight': 0,
    'go_straight': 0,
    'left': 1,
    'turn_left': 1,
    'right': 2,
    'turn_right': 2,
}
ID_TO_DRIVING_COMMAND = {
    0: 'forward',
    1: 'left',
    2: 'right',
}
NUM_DRIVING_COMMANDS = len(ID_TO_DRIVING_COMMAND)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, bool))


def driving_command_to_id(driving_command: Any) -> int:
    """Convert a raw Phase 2 `driving_command` field into a stable integer id."""

    if isinstance(driving_command, bytes):
        driving_command = driving_command.decode('utf-8')

    if hasattr(driving_command, 'item') and not isinstance(driving_command, (str, bytes)):
        try:
            driving_command = driving_command.item()
        except ValueError:
            pass

    if hasattr(driving_command, 'tolist') and not isinstance(driving_command, (str, bytes)):
        driving_command = driving_command.tolist()

    if isinstance(driving_command, str):
        key = driving_command.strip().lower().replace('-', '_').replace(' ', '_')
        try:
            return DRIVING_COMMAND_TO_ID[key]
        except KeyError as exc:
            available = ', '.join(sorted(DRIVING_COMMAND_TO_ID))
            raise ValueError(
                f"Unsupported driving_command string '{driving_command}'. Available aliases: {available}"
            ) from exc

    if isinstance(driving_command, Sequence) and not isinstance(driving_command, (str, bytes)):
        if len(driving_command) == 1:
            return driving_command_to_id(driving_command[0])
        if len(driving_command) == NUM_DRIVING_COMMANDS and all(_is_numeric(value) for value in driving_command):
            return int(max(range(len(driving_command)), key=lambda index: float(driving_command[index])))
        raise ValueError(
            "Unsupported driving_command sequence. Expected a scalar command id, command name, "
            "or a length-3 one-hot / score vector."
        )

    if isinstance(driving_command, bool):
        command_id = int(driving_command)
    elif isinstance(driving_command, int):
        command_id = driving_command
    elif isinstance(driving_command, float):
        if not driving_command.is_integer():
            raise ValueError(f"Non-integer driving_command value: {driving_command}")
        command_id = int(driving_command)
    else:
        raise TypeError(f"Unsupported driving_command type: {type(driving_command)!r}")

    if command_id not in ID_TO_DRIVING_COMMAND:
        raise ValueError(
            f"driving_command id {command_id} is out of range. Expected one of {sorted(ID_TO_DRIVING_COMMAND)}."
        )

    return command_id


def _format_depth_tensor(depth: Any) -> torch.Tensor:
    depth_tensor = torch.as_tensor(depth, dtype=torch.float32)

    if depth_tensor.ndim == 2:
        return depth_tensor.unsqueeze(0)
    if depth_tensor.ndim == 3:
        if depth_tensor.shape[0] == 1:
            return depth_tensor
        if depth_tensor.shape[-1] == 1:
            return depth_tensor.permute(2, 0, 1)

    raise ValueError(
        f"Unsupported depth tensor shape {tuple(depth_tensor.shape)}. Expected [H, W], [H, W, 1], or [1, H, W]."
    )


class DrivingDataset(Dataset):
    """Phase 2 dataset returning camera, history, command, and optional depth supervision."""

    def __init__(
        self,
        file_list: Sequence[str | Path],
        *,
        test: bool = False,
        use_depth_aux: bool = False,
    ):
        self.samples = [Path(path) for path in file_list]
        self.test = test
        self.use_depth_aux = use_depth_aux

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        with self.samples[idx].open('rb') as file:
            data = pickle.load(file)

        batch = {
            'camera': torch.as_tensor(data['camera'], dtype=torch.float32).permute(2, 0, 1),
            'history': torch.as_tensor(data['sdc_history_feature'], dtype=torch.float32),
            'driving_command': torch.tensor(driving_command_to_id(data['driving_command']), dtype=torch.long),
        }

        if self.test:
            # Test/inference intentionally ignores any depth or semantic fields that might be present locally.
            return batch

        batch['future'] = torch.as_tensor(data['sdc_future_feature'], dtype=torch.float32)

        if self.use_depth_aux:
            if 'depth' not in data:
                raise KeyError(
                    f"Depth supervision was requested but sample {self.samples[idx]} has no 'depth' field."
                )
            batch['depth'] = _format_depth_tensor(data['depth'])

        # Semantic supervision is intentionally skipped in this first clean Phase 2 pass.
        return batch
