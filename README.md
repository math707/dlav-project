# DLAV Final Project

Repository for the EPFL DLAV final project. The repo keeps all three phases in one place, with phase-specific notebooks under `notebooks/`, source code under `src/`, and shared infrastructure under `src/shared/`.

## Repository Structure

```text
dlav-project/
|-- notebooks/
|   |-- phase1/
|   |   |-- DLAV_Phase1.ipynb
|   |   `-- README.md
|   |-- phase2/
|   |   |-- DLAV_Phase2.ipynb
|   |   `-- README.md
|   |-- phase3/
|   |   |-- DLAV_Phase3.ipynb
|   |   `-- README.md
|   |-- starter/
|   |   |-- DLAV_Phase2_starter_reference.ipynb
|   |   `-- DLAV_Phase3_starter_reference.ipynb
|   `-- DLAV_Phase1.ipynb
|-- src/
|   |-- shared/
|   |-- phase1/
|   |-- phase2/
|   `-- phase3/
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
|-- data/
|-- requirements.txt
|-- README.md
`-- project_description.md
```

## Phase Overview

| Phase | Focus | Entry Point | Documentation |
| --- | --- | --- | --- |
| Phase 1 | Trajectory prediction from camera and motion history | [notebooks/phase1/DLAV_Phase1.ipynb](notebooks/phase1/DLAV_Phase1.ipynb) | [notebooks/phase1/README.md](notebooks/phase1/README.md) |
| Phase 2 | Perception-aware planning with driving command and depth auxiliary supervision | [notebooks/phase2/DLAV_Phase2.ipynb](notebooks/phase2/DLAV_Phase2.ipynb) | [notebooks/phase2/README.md](notebooks/phase2/README.md) |
| Phase 3 | Sim-to-real generalization from camera + motion history only | [notebooks/phase3/DLAV_Phase3.ipynb](notebooks/phase3/DLAV_Phase3.ipynb) | [notebooks/phase3/README.md](notebooks/phase3/README.md) |

## Common Setup

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the phase notebook you want to run in Jupyter or Colab.
4. Edit the configuration cell near the top of the notebook.
5. Run the notebook from top to bottom.

The notebooks support both local execution and Colab. They can also download the required datasets automatically when `DOWNLOAD_DATA_IF_MISSING = True`.

## Data Layout

Local data is expected under `data/`.

Phase 1 / Phase 2:

- `data/train/`
- `data/val/`
- `data/test_public/`

Phase 3:

- `data/train/`
- `data/val_real/`
- `data/test_public_real/`

These folders are not tracked by git.

## Phase 3 Summary

Phase 3 is the sim-to-real track. The current clean implementation:

- uses only `camera` and `sdc_history_feature`
- predicts future XY positions with output shape `[B, 60, 2]`
- does not use depth labels or semantic labels
- mixes all synthetic `train/` samples with a randomized subset of labeled real `val_real/` samples
- validates on the remaining labeled real samples
- applies photometric augmentations only by default to keep future labels consistent

The Phase 3 planner lives in `src/phase3/model.py` and uses a pretrained ResNet18 backbone plus a small history encoder and fusion MLP.

## Running Phase 3 Training

1. Open [notebooks/phase3/DLAV_Phase3.ipynb](notebooks/phase3/DLAV_Phase3.ipynb).
2. Keep or adjust the default baseline config:
   - `MODEL_NAME = 'phase3_resnet18'`
   - `BATCH_SIZE = 32`
   - `NUM_EPOCHS = 100`
   - `LR = 1e-3`
   - `WEIGHT_DECAY = 1e-4`
   - `REAL_TRAIN_COUNT = 500`
   - `PRETRAINED = True`
   - `BACKBONE_LR_SCALE = 0.1`
   - `BACKBONE_WARMUP_EPOCHS = 2`
3. Run the notebook from top to bottom.

The notebook will:

- bootstrap the repo path locally or in Colab
- download `train/`, `val_real/`, and `test_public_real/` with the starter notebook Google Drive links
- build a randomized real train/validation split with `seed=42` by default
- train with ADE/FDE reporting
- save best and last checkpoints
- create `submission_phase3.csv`

## Generating the Kaggle Submission

Use the last cells of [notebooks/phase3/DLAV_Phase3.ipynb](notebooks/phase3/DLAV_Phase3.ipynb).

By default the notebook:

- reloads the best checkpoint selected by validation ADE
- runs inference on `data/test_public_real/` in sorted numeric filename order
- writes a Kaggle-compatible CSV with columns `id, x_1, y_1, ..., x_60, y_60`

## Outputs

All outputs are phase-scoped.

- Run folders: `outputs/runs/<phase>/<timestamp>_<run_name>/`
- Legacy checkpoint copy: `outputs/checkpoints/<phase>/<phase>_model.pth`
- Legacy submission copy: `outputs/submissions/<phase>/submission_<phase>.csv`

For Phase 3 specifically:

- best checkpoint inside the run: `outputs/runs/phase3/<timestamp>_<run_name>/model.pth`
- last checkpoint inside the run: `outputs/runs/phase3/<timestamp>_<run_name>/model_last.pth`
- run-scoped submission: `outputs/runs/phase3/<timestamp>_<run_name>/submission_phase3.csv`
- legacy best-checkpoint copy: `outputs/checkpoints/phase3/phase3_model.pth`
- legacy submission copy: `outputs/submissions/phase3/submission_phase3.csv`

## Notes

- `src/shared/` contains reusable project setup, logging, run tracking, optimizer/scheduler setup, and dataset download helpers.
- Phase-specific details and recommended settings are documented in the phase README files.
- The Phase 3 training and validation loops consistently optimize XY only, fixing the original starter mismatch between train and validation loss definitions.
