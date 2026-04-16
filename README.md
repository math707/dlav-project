# DLAV Project

End-to-end planner for autonomous driving.

## Project Structure

```text
dlav-project/
├── notebooks/
│   └── DLAV_Phase1.ipynb
├── src/
│   ├── dataset.py
│   ├── logger.py
│   ├── model.py
│   └── train.py
├── outputs/
│   ├── checkpoints/
│   └── submissions/
├── data/                 # local only, not tracked by Git
├── .gitignore
└── README.md
```

Runtime additions not shown in the tree above:

- `src/run_utils.py` centralizes run directory creation and artifact tracking.
- `outputs/runs/` stores timestamped run folders such as `outputs/runs/20260416_113000_baseline/`.

## Data Layout

The notebook expects the following directories inside `data/`:

- `data/train/`
- `data/val/`
- `data/test_public/`

If they are missing, `notebooks/DLAV_Phase1.ipynb` can download and extract the archives into `data/`.

## Workflow

There is a single source of truth for experimentation: `notebooks/DLAV_Phase1.ipynb`.

- In local VS Code, open the notebook from this repository and run it from top to bottom.
- In Google Colab, open the same notebook from GitHub. The setup cells detect Colab, clone the repository into `/content/dlav-project` if needed, install only the missing dependency required for dataset download, and then use the files from the cloned repo.

Each notebook execution creates a primary run folder under `outputs/runs/<timestamp>_<run_name>/`.

- The run folder stores `model.pth`, `submission_phase1.csv`, `metrics.json`, `summary.txt`, and `run.log` when available.
- For backward compatibility, the notebook also keeps historical copies in `outputs/checkpoints/phase1_model.pth` and `outputs/submissions/submission_phase1.csv`.
- In Google Colab, runs are first written to `/content/dlav-project/outputs/runs/`. If Google Drive is already mounted, the notebook also copies the completed run folder to `/content/drive/MyDrive/dlav-project-runs/`.

## Notes

- `data/` is intentionally ignored by Git.
- Generated run artifacts, checkpoints, and submissions are ignored by Git, while the folder structure stays versioned with `.gitkeep`.
- Training and dataset interfaces stay unchanged: models still use `forward(camera, history)` and return `(batch_size, 60, 3)`.
- `src/model.py` keeps the original `DrivingPlanner` baseline and adds `DrivingPlannerModelA` as a first stronger variant.
- For clean comparisons, `src/model.py` also exposes `build_model("baseline")` and `build_model("model_a")`.
