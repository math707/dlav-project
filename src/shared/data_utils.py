import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path


DATASET_SPECS = {
    'train': {
        'file_id': '1YkGwaxBKNiYL2nq--cB6WMmYGzRmRKVr',
        'zip_name': 'dlav_train.zip',
        'target_subdir': 'train',
    },
    'val': {
        'file_id': '1wtmT_vH9mMUNOwrNOMFP6WFw6e8rbOdu',
        'zip_name': 'dlav_val.zip',
        'target_subdir': 'val',
    },
    'test_public': {
        'file_id': '1G9xGE7s-Ikvvc2-LZTUyuzhWAlNdLTLV',
        'zip_name': 'dlav_test_public.zip',
        'target_subdir': 'test_public',
    },
}


def sorted_pkl_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return sorted(directory.glob('*.pkl'), key=lambda path: int(path.stem))


def has_pkl_files(directory: str | Path) -> bool:
    directory = Path(directory)
    return directory.is_dir() and any(directory.glob('*.pkl'))


def ensure_gdown(in_colab: bool):
    if importlib.util.find_spec('gdown') is None:
        if not in_colab:
            raise ImportError(
                'gdown is required only for automatic dataset download. '
                'Install it locally or place the archives/files manually under data/.'
            )
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'])

    import gdown

    return gdown


def ensure_dataset(
    name: str,
    file_id: str,
    zip_name: str,
    data_dir: str | Path,
    target_dir: str | Path,
    in_colab: bool,
    project_root: str | Path | None = None,
) -> Path:
    data_dir = Path(data_dir)
    target_dir = Path(target_dir)
    project_root = Path(project_root).resolve() if project_root is not None else None
    zip_path = data_dir / zip_name

    def _display_path(path: Path) -> Path:
        if project_root is not None:
            try:
                return path.relative_to(project_root)
            except ValueError:
                return path
        return path

    if has_pkl_files(target_dir):
        print(f'{name}: found extracted files in {_display_path(target_dir)}')
        return target_dir

    if not zip_path.exists():
        gdown = ensure_gdown(in_colab)
        download_url = f'https://drive.google.com/uc?id={file_id}'
        print(f'{name}: downloading {zip_name}...')
        gdown.download(download_url, str(zip_path), quiet=False)
    else:
        print(f'{name}: found archive {zip_name}, skipping download.')

    print(f'{name}: extracting into {_display_path(data_dir)}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    if not has_pkl_files(target_dir):
        raise FileNotFoundError(f'{name}: no .pkl files found in {target_dir}')

    print(f'{name}: ready in {_display_path(target_dir)}')
    return target_dir


def ensure_all_datasets(
    data_dir: str | Path,
    in_colab: bool,
    dataset_specs: dict | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    data_dir = Path(data_dir)
    dataset_specs = dataset_specs or DATASET_SPECS

    resolved_targets: dict[str, Path] = {}
    for name, config in dataset_specs.items():
        target_dir = data_dir / config['target_subdir']
        resolved_targets[name] = ensure_dataset(
            name=name,
            file_id=config['file_id'],
            zip_name=config['zip_name'],
            data_dir=data_dir,
            target_dir=target_dir,
            in_colab=in_colab,
            project_root=project_root,
        )

    return resolved_targets
