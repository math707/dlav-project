# DLAV Project

End-to-end planner for the DLAV course project. The repository is organized so that:

- the notebook stays the main entry point for running experiments,
- the core logic lives in `src/`,
- the workflow remains easy to use in Google Colab,
- run artifacts are saved in a predictable way for comparison and submission.

## Quick Start

1. Open `notebooks/DLAV_Phase1.ipynb` from this repository, preferably in Google Colab.
2. Edit the first code cell at the top of the notebook to choose the run parameters such as `RUN_NAME`, `MODEL_NAME`, `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE_NAME`, and `WEIGHT_DECAY`.
3. Run all notebook cells from top to bottom.
4. Check the run folder created under `outputs/runs/<timestamp>_<run_name>/`.
5. The main artifacts are saved there: `model.pth` for the best checkpoint, `model_last.pth` for the last epoch, and `submission_phase1.csv` for the generated submission.

## Expected First-Time Workflow

For a TA or assistant landing on the repository for the first time, the recommended path is:

1. Open `notebooks/DLAV_Phase1.ipynb` in Colab.
2. Read the short notebook introduction and adjust the parameters in the top configuration cell.
3. Run all cells once without changing the project code in `src/`.
4. Inspect `outputs/runs/<timestamp>_<run_name>/` for the checkpoint, metrics, logs, and submission file.

## Repository Structure

```text
dlav-project/
|-- notebooks/
|   `-- DLAV_Phase1.ipynb
|-- src/
|   |-- __init__.py
|   |-- data_utils.py
|   |-- dataset.py
|   |-- logger.py
|   |-- model.py
|   |-- project_setup.py
|   |-- run_utils.py
|   |-- submission.py
|   |-- train.py
|   `-- training_setup.py
|-- outputs/
|   |-- checkpoints/
|   |-- runs/
|   `-- submissions/
|-- data/                     # local / Colab data, not tracked by git
|-- requirements.txt
|-- README.md
`-- project_description.pdf
```

## What Lives Where

- `notebooks/DLAV_Phase1.ipynb`
  Main run notebook. It is intended for environment setup, Colab/Drive setup, experiment configuration, training, inference, and quick result inspection.
- `src/model.py`
  Model definitions. It keeps the original baseline and the improved `Model A`, plus `build_model(...)` for easy switching.
- `src/dataset.py`
  Dataset loading logic.
- `src/train.py`
  Training loop and validation metrics.
- `src/training_setup.py`
  Optimizer and scheduler construction.
- `src/submission.py`
  Prediction and submission CSV generation.
- `src/run_utils.py`
  Run directory creation, metrics saving, summaries, and artifact syncing.
- `src/project_setup.py`
  Project root detection and Colab / Google Drive helpers.
- `src/data_utils.py`
  Dataset download, extraction, and file discovery helpers.
- `outputs/runs/`
  Timestamped run folders with checkpoints, metrics, logs, and submissions.

## Main Workflow

The recommended workflow is to use `notebooks/DLAV_Phase1.ipynb` as the single entry point, edit the parameter cell at the top, run all cells, and then inspect the generated run folder under `outputs/runs/`.

The notebook is intentionally run-oriented. Reusable logic has been moved into `src/` so the notebook stays easier to read and easier for a TA or assistant to follow.

## Notebook Parameters

The first code cell exposes the main run parameters directly, without a heavy config system. The most important ones are:

- `RUN_NAME`
- `MODEL_NAME`
- `BATCH_SIZE`
- `NUM_EPOCHS`
- `LEARNING_RATE_NAME`
- `LEARNING_RATE`
- `WEIGHT_DECAY`
- `USE_LR_SCHEDULER`
- `SCHEDULER_NAME`
- `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE`
- `DOWNLOAD_DATA_IF_MISSING`
- `SYNC_RUN_TO_DRIVE`

This makes it easy to compare runs while keeping the workflow simple.

## Running In Google Colab

Colab is the recommended setup for this project.

### What the notebook does in Colab

- Detects that it is running in Colab.
- Clones or updates the repository into `/content/dlav-project` when needed.
- Optionally mounts Google Drive.
- Uses Google Drive as a backup location for completed run folders when available.
- Downloads the dataset automatically if it is missing and `DOWNLOAD_DATA_IF_MISSING = True`.

### Typical Colab usage

1. Open `notebooks/DLAV_Phase1.ipynb` in Colab.
2. Set the editable parameters in the first code cell.
3. Keep `MOUNT_DRIVE_IN_COLAB = True` if you want run artifacts copied to Drive.
4. Run all cells.

By default, completed runs can be backed up to:

`/content/drive/MyDrive/dlav-project-runs/`

This is useful because Colab VM storage is temporary.

## Running Locally

If you want to run locally:

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure the dataset is available under:
   - `data/train/`
   - `data/val/`
   - `data/test_public/`
4. Open `notebooks/DLAV_Phase1.ipynb` from inside the repository.
5. Run the notebook from top to bottom.

The notebook detects the project root automatically when opened from the repository.

## Models

The project keeps multiple model variants in the same repository for clean comparisons.

- `DrivingPlanner`
  Original baseline architecture from the starter notebook.
- `DrivingPlannerModelA`
  First improved model variant with a stronger CNN encoder, compact history MLP, and better fusion head.
- `DrivingPlannerModelB`
  Comparison-friendly next-step variant that replaces the custom visual encoder with a pretrained ResNet18 backbone, keeps a simple history MLP, and predicts the same `(batch_size, 60, 3)` trajectory output.

### Model Architecture

The model takes two inputs:

- a camera image tensor,
- a motion/history tensor describing the recent ego-vehicle trajectory.

The baseline model uses a very shallow CNN to extract image features, flattens the history tensor, concatenates both branches, and maps them directly to the future trajectory with a single linear decoder.

`Model A` keeps the same overall input/output structure but improves the internal feature extraction. It uses a stronger CNN visual encoder with progressive downsampling, then applies global average pooling to produce a compact visual embedding. The history branch is encoded with a small MLP, the two embeddings are fused with another small MLP, and a final prediction head outputs the trajectory.

All models return a predicted future trajectory with shape `(batch_size, 60, 3)`. Because `Model A` and `Model B` preserve the same `forward(camera, history)` interface and output format as the baseline, comparisons between variants remain clean and direct.

Use the notebook parameter:

```python
MODEL_NAME = "baseline"   # or "model_a" or "model_b"
```

Internally, model creation goes through:

```python
model = build_model(MODEL_NAME)
```

This keeps comparisons simple and avoids overwriting old architectures.

`Model B` also supports a smaller learning rate on the pretrained backbone through the existing optimizer helper:

```python
optimizer = build_optimizer(
    model,
    learning_rate=3e-4,
    weight_decay=1e-4,
    backbone_lr_scale=0.1,
)
```

Optional early stopping based on validation ADE can be enabled directly in the training call:

```python
TRAINING_SUMMARY = train(
    model,
    train_loader,
    val_loader,
    optimizer,
    logger,
    scheduler=scheduler,
    scheduler_metric="val_ADE",
    best_checkpoint_path=RUN_CONTEXT.checkpoint_path,
    early_stopping_patience=6,
    early_stopping_min_delta=1e-3,
)
```

## Training, Checkpoints, And Outputs

Each execution creates a run directory:

```text
outputs/runs/<timestamp>_<run_name>/
```

A run folder typically contains:

- `model.pth`
  Best checkpoint of the run, selected by validation ADE.
- `model_last.pth`
  Last-epoch checkpoint.
- `metrics.json`
  Structured run metadata and recorded metrics.
- `summary.txt`
  Human-readable run summary.
- `run.log`
  Training log.
- `submission_phase1.csv`
  Generated submission file.

For backward compatibility, the notebook also writes:

- `outputs/checkpoints/phase1_model.pth`
- `outputs/submissions/submission_phase1.csv`

These are legacy convenience copies. The main source of truth is the timestamped run folder.

## Best Checkpoint Behavior

The training setup now tracks the best validation ADE during training.

- The best checkpoint is saved to `model.pth`.
- The last model is saved separately to `model_last.pth`.
- By default, the notebook reloads the best checkpoint before qualitative evaluation and submission generation.

This behavior is controlled by:

```python
RELOAD_BEST_CHECKPOINT_FOR_INFERENCE = True
```

So, by default, the submission CSV is generated from the best checkpoint of the run, not just the last epoch.

## Inference And Submission Generation

Inference and submission generation are launched from the notebook and implemented through helpers in `src/submission.py`.

The output CSV is saved to:

```text
outputs/runs/<timestamp>_<run_name>/submission_phase1.csv
```

and also copied to:

```text
outputs/submissions/submission_phase1.csv
```

The expected output format remains unchanged.

## Reproducibility And Comparison

The repository is organized to make experiments easier to compare:

- baseline and improved models live side by side,
- key hyperparameters are visible at the top of the notebook,
- each run gets its own timestamped folder,
- metrics, logs, and checkpoints are grouped together,
- the best validation checkpoint is preserved automatically.

This is meant to keep the project easy to inspect for course staff while still being practical for ongoing experimentation.

## Git And Submission Notes

- `data/` is ignored by git.
- generated checkpoints, submissions, and run artifacts are ignored by git.
- folder placeholders are kept with `.gitkeep`.
- the notebook and `src/` code are the files that matter most for the submission.

Before pushing, make sure the repository contains code and documentation, not local datasets or temporary outputs.
