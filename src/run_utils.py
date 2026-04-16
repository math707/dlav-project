import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_COLAB_DRIVE_ROOT = Path('/content/drive/MyDrive/dlav-project-runs')


@dataclass(frozen=True)
class RunContext:
    run_name: str
    timestamp: str
    environment: str
    project_root: Path
    outputs_dir: Path
    runs_root: Path
    run_dir: Path
    checkpoint_path: Path
    submission_path: Path
    metrics_path: Path
    summary_path: Path
    log_path: Path
    drive_root: Path | None = None
    drive_run_dir: Path | None = None

    @property
    def drive_backup_enabled(self) -> bool:
        return self.drive_run_dir is not None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


def sanitize_run_name(run_name: str | None, default_name: str = 'baseline') -> str:
    candidate = (run_name or default_name).strip().lower()
    candidate = re.sub(r'[^a-z0-9._-]+', '_', candidate)
    candidate = candidate.strip('._-')
    return candidate or default_name


def detect_environment(in_colab: bool) -> str:
    return 'colab' if in_colab else 'local'


def is_google_drive_mounted() -> bool:
    return Path('/content/drive/MyDrive').is_dir()


def create_run_context(
    project_root: str | Path,
    in_colab: bool,
    run_name: str | None = None,
    default_run_name: str = 'baseline',
    drive_root: str | Path | None = None,
) -> RunContext:
    project_root = Path(project_root).resolve()
    outputs_dir = project_root / 'outputs'
    runs_root = outputs_dir / 'runs'
    runs_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    clean_run_name = sanitize_run_name(run_name, default_name=default_run_name)
    run_dir = runs_root / f'{timestamp}_{clean_run_name}'
    suffix = 1
    while run_dir.exists():
        run_dir = runs_root / f'{timestamp}_{clean_run_name}_{suffix:02d}'
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    environment = detect_environment(in_colab)
    resolved_drive_root: Path | None = None
    drive_run_dir: Path | None = None

    if environment == 'colab' and is_google_drive_mounted():
        resolved_drive_root = Path(drive_root or DEFAULT_COLAB_DRIVE_ROOT)
        drive_run_dir = resolved_drive_root / run_dir.name

    return RunContext(
        run_name=clean_run_name,
        timestamp=timestamp,
        environment=environment,
        project_root=project_root,
        outputs_dir=outputs_dir,
        runs_root=runs_root,
        run_dir=run_dir,
        checkpoint_path=run_dir / 'model.pth',
        submission_path=run_dir / 'submission_phase1.csv',
        metrics_path=run_dir / 'metrics.json',
        summary_path=run_dir / 'summary.txt',
        log_path=run_dir / 'run.log',
        drive_root=resolved_drive_root,
        drive_run_dir=drive_run_dir,
    )


def _normalize_metrics_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _normalize_metrics_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_metrics_value(item) for item in value]
    return value


def build_metrics_payload(run_context: RunContext, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'run_name': run_context.run_name,
        'timestamp': run_context.timestamp,
        'environment': run_context.environment,
        'run_dir': str(run_context.run_dir),
        'checkpoint_path': None,
        'submission_path': None,
    }

    if metrics:
        payload.update(_normalize_metrics_value(metrics))

    return payload


def save_metrics(run_context: RunContext, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = build_metrics_payload(run_context, metrics)
    run_context.metrics_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload


def write_summary(run_context: RunContext, metrics: dict[str, Any] | None = None) -> str:
    payload = build_metrics_payload(run_context, metrics)
    lines = [
        f"run_name: {payload.get('run_name')}",
        f"timestamp: {payload.get('timestamp')}",
        f"environment: {payload.get('environment')}",
        f"device: {payload.get('device')}",
        f"batch_size: {payload.get('batch_size')}",
        f"learning_rate_name: {payload.get('learning_rate_name')}",
        f"learning_rate: {payload.get('learning_rate')}",
        f"weight_decay: {payload.get('weight_decay')}",
        f"scheduler_enabled: {payload.get('scheduler_enabled')}",
        f"scheduler_name: {payload.get('scheduler_name')}",
        f"scheduler_metric: {payload.get('scheduler_metric')}",
        f"num_epochs: {payload.get('num_epochs')}",
        f"train_loss_final: {payload.get('train_loss_final')}",
        f"val_loss_final: {payload.get('val_loss_final')}",
        f"val_ADE_final: {payload.get('val_ADE_final')}",
        f"val_FDE_final: {payload.get('val_FDE_final')}",
        f"best_val_ADE: {payload.get('best_val_ADE')}",
        f"best_val_ADE_epoch: {payload.get('best_val_ADE_epoch')}",
        f"best_checkpoint_path: {payload.get('best_checkpoint_path')}",
        f"last_checkpoint_path: {payload.get('last_checkpoint_path')}",
        f"checkpoint_path: {payload.get('checkpoint_path')}",
        f"submission_path: {payload.get('submission_path')}",
    ]
    content = '\n'.join(lines) + '\n'
    run_context.summary_path.write_text(content, encoding='utf-8')
    return content


def copy_artifact_to_destination(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return destination_path


def sync_run_to_drive(run_context: RunContext) -> Path | None:
    if run_context.environment != 'colab' or not is_google_drive_mounted():
        return None

    drive_root = run_context.drive_root or DEFAULT_COLAB_DRIVE_ROOT
    drive_run_dir = run_context.drive_run_dir or (drive_root / run_context.run_dir.name)

    drive_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_context.run_dir, drive_run_dir, dirs_exist_ok=True)
    return drive_run_dir
