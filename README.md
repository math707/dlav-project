# DLAV Project

End-to-end planning repository for the EPFL DLAV final project. The repo is now organized for the full course submission requirement: all three phases live in the same repository under separate folders, while shared utilities stay centralized in `src/shared/`.

The official submission repository is the private GitHub Classroom repo:
`https://github.com/vita-student-projects-2026/final-project-math.git`

## Current Status

- Phase 1 is implemented and remains the current working pipeline.
- Phase 2 has been structurally prepared but is not yet re-implemented in the cleaned `src/phase2/` layout.
- Phase 3 has placeholder folders so the final repository structure is already in place.

This refactor preserves the current Phase 1 model and training behavior. The goal of this step is structural cleanup only.

## Repository Structure

```text
dlav-project/
|-- notebooks/
|   |-- phase1/
|   |   `-- DLAV_Phase1.ipynb
|   |-- phase2/
|   |   `-- DLAV_Phase2.ipynb
|   |-- phase3/
|   |   `-- DLAV_Phase3.ipynb
|   |-- starter/
|   |   `-- DLAV_Phase2_starter_reference.ipynb
|   `-- DLAV_Phase1.ipynb                  # temporary compatibility copy
|-- src/
|   |-- shared/
|   |   |-- data_utils.py
|   |   |-- logger.py
|   |   |-- project_setup.py
|   |   |-- run_utils.py
|   |   |-- submission.py
|   |   `-- training_setup.py
|   |-- phase1/
|   |   |-- dataset.py
|   |   |-- model.py
|   |   `-- train.py
|   |-- phase2/
|   |   `-- __init__.py
|   |-- phase3/
|   |   `-- __init__.py
|   |-- dataset.py                         # backward-compatible wrapper
|   |-- model.py                           # backward-compatible wrapper
|   |-- train.py                           # backward-compatible wrapper
|   |-- data_utils.py                      # backward-compatible wrapper
|   |-- logger.py                          # backward-compatible wrapper
|   |-- project_setup.py                   # backward-compatible wrapper
|   |-- run_utils.py                       # backward-compatible wrapper
|   |-- submission.py                      # backward-compatible wrapper
|   `-- training_setup.py                  # backward-compatible wrapper
|-- outputs/
|   |-- runs/
|   |   |-- phase1/
|   |   |-- phase2/
|   |   `-- phase3/
|   |-- checkpoints/
|   |   |-- phase1/
|   |   |-- phase2/
|   |   `-- phase3/
|   `-- submissions/
|       |-- phase1/
|       |-- phase2/
|       `-- phase3/
|-- data/                                  # local / Colab data, not tracked by git
|-- requirements.txt
|-- README.md
`-- project_description.md
```

## Where Each Phase Lives

- Phase 1 notebook: `notebooks/phase1/DLAV_Phase1.ipynb`
- Phase 1 code: `src/phase1/`
- Phase 2 cleaned notebook placeholder: `notebooks/phase2/DLAV_Phase2.ipynb`
- Phase 2 starter reference notebook: `notebooks/starter/DLAV_Phase2_starter_reference.ipynb`
- Phase 2 future code location: `src/phase2/`
- Phase 3 notebook placeholder: `notebooks/phase3/DLAV_Phase3.ipynb`
- Phase 3 future code location: `src/phase3/`
- Shared helpers for all phases: `src/shared/`

## Running Phase 1

The recommended Phase 1 entry point is now:

`notebooks/phase1/DLAV_Phase1.ipynb`

The older root-level notebook path `notebooks/DLAV_Phase1.ipynb` is still kept temporarily as a compatibility copy during the transition.

### Local workflow

1. Clone the private GitHub Classroom repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure the dataset is available under:
   - `data/train/`
   - `data/val/`
   - `data/test_public/`
4. Open `notebooks/phase1/DLAV_Phase1.ipynb`.
5. Edit the parameter cell at the top if needed.
6. Run all cells from top to bottom.

### Colab workflow

The notebook still supports Colab. It detects the repository root, can clone/update the private Classroom repo under `/content/dlav-project`, can mount Google Drive, and can download the dataset automatically when `DOWNLOAD_DATA_IF_MISSING = True`.

## Phase 1 Outputs

Phase 1 now writes phase-scoped artifacts under:

- `outputs/runs/phase1/<timestamp>_<run_name>/`
- `outputs/checkpoints/phase1/phase1_model.pth`
- `outputs/submissions/phase1/submission_phase1.csv`

Inside each run directory, the main artifacts remain:

- `model.pth` for the best checkpoint
- `model_last.pth` for the last epoch
- `metrics.json` for structured metadata
- `summary.txt` for a short run summary
- `run.log` for training logs
- `submission_phase1.csv` for the generated submission

The Phase 1 best-checkpoint selection, reload behavior, scheduler support, early stopping, and submission generation flow are unchanged by this refactor.

## Phase 1 Models

Phase 1 still exposes the same model registry through `src.phase1.model.build_model(...)`:

- `baseline`
- `model_a`
- `model_b`
- `model_b_v2`

The recommended configuration remains `model_b_v2`.

## Shared vs Phase-Specific Code

- `src/shared/` contains reusable infrastructure: project bootstrap, dataset download helpers, logging, run tracking, submission generation, and optimizer/scheduler setup.
- `src/phase1/` contains the current Phase 1 dataset, model, and training loop.
- The old top-level `src/*.py` entry points now exist only as backward-compatible wrappers and should not be used for new phase-specific development.

## Phase 2 And Phase 3 Integration Status

Phase 2 and Phase 3 are intentionally split into their own folders now so the final repo stays organized as the project grows.

- `notebooks/phase2/DLAV_Phase2.ipynb` is a cleaned placeholder notebook.
- `notebooks/starter/DLAV_Phase2_starter_reference.ipynb` preserves the raw starter notebook for reference.
- The legacy root-level `notebooks/DLAV_Phase2.ipynb` duplicate is intentionally not part of the cleaned layout.
- `src/phase2/` is reserved for the cleaned Phase 2 implementation that will be integrated next.
- `notebooks/phase3/DLAV_Phase3.ipynb` and `src/phase3/` are placeholders for the final milestone.

## Notes

- `data/` is ignored by git.
- Generated outputs under `outputs/` are ignored by git; only `.gitkeep` placeholders are tracked.
- The repository root detection still works from nested notebook folders because the setup code searches parent directories for the project root.
- `project_description.md` is the current project brief in this repository.
