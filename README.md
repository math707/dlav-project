# DLAV Final Project

Repository for the EPFL DLAV final project. The final submission keeps all three phases in one repo, with phase-specific notebooks and code under separate folders and shared utilities under `src/shared/`.

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
|   |   `-- DLAV_Phase2_starter_reference.ipynb
|   `-- DLAV_Phase1.ipynb              # temporary compatibility copy
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
| Phase 3 | Sim-to-real generalization placeholder | [notebooks/phase3/DLAV_Phase3.ipynb](notebooks/phase3/DLAV_Phase3.ipynb) | [notebooks/phase3/README.md](notebooks/phase3/README.md) |

## Common Setup

1. Clone the private GitHub Classroom repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure the dataset is available under:
   - `data/train/`
   - `data/val/`
   - `data/test_public/`
4. Open the phase notebook you want to run in Jupyter or Colab.
5. Edit the configuration cell at the top of that notebook.
6. Run the notebook from top to bottom.

The notebooks support both local execution and Colab. They can also download the dataset automatically when `DOWNLOAD_DATA_IF_MISSING = True`.

## Data Layout

Local data is expected under `data/`:

- `data/train/`
- `data/val/`
- `data/test_public/`

These folders are not tracked by git.

## Outputs

All outputs are phase-scoped:

- Run folders: `outputs/runs/<phase>/<timestamp>_<run_name>/`
- Legacy checkpoint copy: `outputs/checkpoints/<phase>/<phase>_model.pth`
- Legacy submission copy: `outputs/submissions/<phase>/submission_<phase>.csv`

Each run folder stores the main artifacts for that notebook run, including the best checkpoint, metrics, summary, log, and submission file.

## Current Recommended Entry Points

- Phase 1: [notebooks/phase1/DLAV_Phase1.ipynb](notebooks/phase1/DLAV_Phase1.ipynb)
- Phase 2: [notebooks/phase2/DLAV_Phase2.ipynb](notebooks/phase2/DLAV_Phase2.ipynb)
- Phase 3: [notebooks/phase3/DLAV_Phase3.ipynb](notebooks/phase3/DLAV_Phase3.ipynb) placeholder

## Notes

- `src/shared/` contains reusable utilities for project setup, logging, run tracking, optimizer/scheduler setup, and submission writing.
- Phase-specific details, model variants, and recommended settings are documented in the phase README files linked above.
- `notebooks/starter/DLAV_Phase2_starter_reference.ipynb` is kept as a reference copy of the original Phase 2 starter notebook.
