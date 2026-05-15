# DLAV Final Project

Repository for the EPFL DLAV final project on trajectory prediction across three phases. The main entry points are the phase notebooks in `notebooks/`, with shared training utilities in `src/shared/` and phase-specific code in `src/phase1/`, `src/phase2/`, and `src/phase3/`.

## Repository Structure

```text
dlav-project/
├─ notebooks/
│  ├─ phase1/        # main Phase 1 notebook + README
│  ├─ phase2/        # main Phase 2 notebook + README
│  ├─ phase3/        # main Phase 3 notebook + README
│  └─ starter/       # archived starter reference notebooks
├─ src/
│  ├─ shared/        # setup, logging, run management, training helpers
│  ├─ phase1/
│  ├─ phase2/
│  └─ phase3/
├─ outputs/          # local run folders, checkpoints, submissions
├─ data/             # datasets, not tracked by git
└─ requirements.txt
```

## Phases

| Phase | Focus | Notebook | Phase README |
| --- | --- | --- | --- |
| Phase 1 | Trajectory prediction from camera and motion history | [DLAV_Phase1.ipynb](notebooks/phase1/DLAV_Phase1.ipynb) | [Phase 1 README](notebooks/phase1/README.md) |
| Phase 2 | Perception-aware planning with command input and depth auxiliary supervision | [DLAV_Phase2.ipynb](notebooks/phase2/DLAV_Phase2.ipynb) | [Phase 2 README](notebooks/phase2/README.md) |
| Phase 3 | Sim-to-real generalization from camera and motion history only | [DLAV_Phase3.ipynb](notebooks/phase3/DLAV_Phase3.ipynb) | [Phase 3 README](notebooks/phase3/README.md) |

## Common Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Put the datasets under `data/`, or keep `DOWNLOAD_DATA_IF_MISSING = True` in the notebook.
3. Open the notebook for the phase you want to run.
4. Edit the top configuration cell.
5. Run the notebook from top to bottom.

Expected data layout:

- Phase 1 / Phase 2: `data/train/`, `data/val/`, `data/test_public/`
- Phase 3: `data/train/`, `data/val_real/`, `data/test_public_real/`

## Train / Reload / Submit Workflow

- Training and validation are handled inside each phase notebook.
- With `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE = True`, the notebook reloads the checkpoint selected by validation ADE before inference.
- Submission generation is done in the final notebook cells and writes a phase-specific CSV.
- Detailed model choices and recommended defaults are documented in the phase READMEs.

## Outputs

- Run folder: `outputs/runs/<phase>/<timestamp>_<run_name>/`
- Legacy checkpoint copy: `outputs/checkpoints/<phase>/`
- Legacy submission copy: `outputs/submissions/<phase>/`

Each run folder contains the best checkpoint (`model.pth`), the last checkpoint (`model_last.pth` when available), and the generated submission CSV.
