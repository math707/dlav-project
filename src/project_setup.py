from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import sys


DEFAULT_REPO_URL = 'https://github.com/math707/dlav-project.git'
DEFAULT_COLAB_PROJECT_DIR = Path('/content/dlav-project')
DEFAULT_COLAB_DRIVE_ROOT = Path('/content/drive/MyDrive/dlav-project-runs')


@dataclass(frozen=True)
class ProjectContext:
    in_colab: bool
    drive_mounted: bool
    project_root: Path
    notebooks_dir: Path
    src_dir: Path
    data_dir: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path
    outputs_dir: Path
    runs_dir: Path
    checkpoints_dir: Path
    submissions_dir: Path
    legacy_checkpoint_path: Path
    legacy_submission_path: Path
    drive_runs_root: Path | None = None


def is_running_in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except ImportError:
        return False


def find_project_root(start: str | Path) -> Path | None:
    start_path = Path(start).resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / 'src').is_dir() and (candidate / 'notebooks').is_dir():
            return candidate
    return None


def ensure_colab_repo(
    repo_url: str = DEFAULT_REPO_URL,
    target_dir: str | Path = DEFAULT_COLAB_PROJECT_DIR,
) -> Path:
    target_path = Path(target_dir)
    git_dir = target_path / '.git'

    if git_dir.is_dir():
        print(f'Updating repository in {target_path}...')
        subprocess.check_call(['git', '-C', str(target_path), 'pull', '--ff-only'])
        return target_path

    if target_path.exists():
        if (target_path / 'src').is_dir() and (target_path / 'notebooks').is_dir():
            print(f'Using existing project directory in {target_path}...')
            return target_path
        raise FileExistsError(
            f'{target_path} exists but is not a Git repository or recognized project root.'
        )

    print(f'Cloning repository into {target_path}...')
    subprocess.check_call(['git', 'clone', repo_url, str(target_path)])
    return target_path


def add_project_root_to_pythonpath(project_root: str | Path) -> Path:
    resolved_root = Path(project_root).resolve()
    if str(resolved_root) not in sys.path:
        sys.path.insert(0, str(resolved_root))
    return resolved_root


def mount_google_drive_if_needed(
    in_colab: bool,
    mount_drive_in_colab: bool,
    mount_point: str = '/content/drive',
) -> bool:
    if not in_colab or not mount_drive_in_colab:
        return False

    drive_root = Path(mount_point) / 'MyDrive'
    if drive_root.is_dir():
        return True

    from google.colab import drive  # type: ignore

    drive.mount(mount_point)
    return drive_root.is_dir()


def prepare_project_context(
    project_root: str | Path,
    in_colab: bool,
    mount_drive_in_colab: bool = False,
    drive_runs_root: str | Path | None = None,
) -> ProjectContext:
    resolved_root = Path(project_root).resolve()
    add_project_root_to_pythonpath(resolved_root)

    drive_mounted = mount_google_drive_if_needed(in_colab, mount_drive_in_colab)

    notebooks_dir = resolved_root / 'notebooks'
    src_dir = resolved_root / 'src'
    data_dir = resolved_root / 'data'
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test_public'
    outputs_dir = resolved_root / 'outputs'
    runs_dir = outputs_dir / 'runs'
    checkpoints_dir = outputs_dir / 'checkpoints'
    submissions_dir = outputs_dir / 'submissions'

    for path in (data_dir, train_dir, val_dir, test_dir, outputs_dir, runs_dir, checkpoints_dir, submissions_dir):
        path.mkdir(parents=True, exist_ok=True)

    resolved_drive_runs_root: Path | None = None
    if in_colab and drive_mounted:
        resolved_drive_runs_root = Path(drive_runs_root or DEFAULT_COLAB_DRIVE_ROOT)

    return ProjectContext(
        in_colab=in_colab,
        drive_mounted=drive_mounted,
        project_root=resolved_root,
        notebooks_dir=notebooks_dir,
        src_dir=src_dir,
        data_dir=data_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        outputs_dir=outputs_dir,
        runs_dir=runs_dir,
        checkpoints_dir=checkpoints_dir,
        submissions_dir=submissions_dir,
        legacy_checkpoint_path=checkpoints_dir / 'phase1_model.pth',
        legacy_submission_path=submissions_dir / 'submission_phase1.csv',
        drive_runs_root=resolved_drive_runs_root,
    )


def change_working_directory(project_root: str | Path) -> Path:
    resolved_root = Path(project_root).resolve()
    os.chdir(resolved_root)
    return resolved_root
