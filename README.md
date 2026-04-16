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

Checkpoints are written to `outputs/checkpoints/`.
Submission files are written to `outputs/submissions/`.

## Notes

- `data/` is intentionally ignored by Git.
- Generated checkpoints and submissions are ignored by Git, while the folder structure stays versioned with `.gitkeep`.
- The current refactor keeps the existing model architecture and training logic in `src/` unchanged.
