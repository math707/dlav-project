# DLAV Project

End-to-end planner for the DLAV course project. The repository is organized so that:

- the notebook stays the main entry point for running experiments,
- the core logic lives in `src/`,
- the workflow remains easy to use in Google Colab,
- run artifacts are saved in a predictable way for comparison and submission.

The current recommended experiment setup uses `model_b_v2`, a pretrained ResNet18-based planner with a small backbone learning rate, a short backbone warmup, scheduler support, and early stopping.

## Quick Start

1. Open `notebooks/DLAV_Phase1.ipynb` from this repository, preferably in Google Colab.
2. Edit the first code cell at the top of the notebook to choose the run parameters. The notebook defaults already point to the current recommended `model_b_v2` setup.
3. Run all notebook cells from top to bottom.
4. Check the run folder created under `outputs/runs/<timestamp>_<run_name>/`.
5. The main artifacts are saved there: `model.pth` for the best checkpoint, `model_last.pth` for the last epoch, and `submission_phase1.csv` for the generated submission.

## Expected First-Time Workflow

For a TA or assistant landing on the repository for the first time, the recommended path is:

1. Open `notebooks/DLAV_Phase1.ipynb` in Colab.
2. Read the short notebook introduction and adjust the parameters in the top configuration cell. If you just want the current recommended run, the defaults are a good starting point.
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
  Model definitions and the `build_model(...)` registry. It contains `baseline`, `model_a`, `model_b`, and `model_b_v2`.
- `src/dataset.py`
  Dataset loading logic.
- `src/train.py`
  Training loop, validation metrics, best-checkpoint tracking, optional early stopping, and optional backbone warmup.
- `src/training_setup.py`
  Optimizer and scheduler construction, including split learning rates for pretrained ResNet18 variants.
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
- `BACKBONE_LEARNING_RATE`
- `BACKBONE_LR_SCALE`
- `BACKBONE_WARMUP_EPOCHS`
- `USE_LR_SCHEDULER`
- `SCHEDULER_NAME`
- `SCHEDULER_METRIC`
- `SCHEDULER_FACTOR`
- `SCHEDULER_PATIENCE`
- `SCHEDULER_MIN_LR`
- `EARLY_STOPPING_PATIENCE`
- `EARLY_STOPPING_MIN_DELTA`
- `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE`
- `DOWNLOAD_DATA_IF_MISSING`
- `SYNC_RUN_TO_DRIVE`

This makes it easy to compare runs while keeping the workflow simple.

Current recommended configuration:

```python
MODEL_NAME = "model_b_v2"
BATCH_SIZE = 32
NUM_EPOCHS = 10000
LEARNING_RATE_NAME = "default"
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BACKBONE_LEARNING_RATE = None
BACKBONE_LR_SCALE = None      # model_b_v2 defaults to 0.1 if left as None
BACKBONE_WARMUP_EPOCHS = 2
USE_LR_SCHEDULER = True
SCHEDULER_NAME = "plateau"
SCHEDULER_METRIC = "val_ADE"
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 6
SCHEDULER_MIN_LR = 1e-5
EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_MIN_DELTA = 1e-3
RELOAD_BEST_CHECKPOINT_FOR_INFERENCE = True
```

`NUM_EPOCHS = 10000` is used as a ceiling for long runs. In practice, the scheduler and early stopping usually end training much earlier once validation ADE stops improving.

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
  First pretrained ResNet18 variant. It keeps the same interface as the earlier models and adds ImageNet-style camera preprocessing inside the model.
- `DrivingPlannerModelBV2`
  Current recommended model. It keeps the `model_b` architecture simple and comparison-friendly, but makes pretrained use safer and fine-tuning cleaner through stricter pretrained-weight loading, a default `0.1x` backbone learning-rate scale, and optional frozen-backbone warmup.

### Model Architecture

The model takes two inputs:

- a camera image tensor,
- a motion/history tensor describing the recent ego-vehicle trajectory.

The baseline model uses a very shallow CNN to extract image features, flattens the history tensor, concatenates both branches, and maps them directly to the future trajectory with a single linear decoder.

`Model A` keeps the same overall input/output structure but improves the internal feature extraction. It uses a stronger CNN visual encoder with progressive downsampling, then applies global average pooling to produce a compact visual embedding. The history branch is encoded with a small MLP, the two embeddings are fused with another small MLP, and a final prediction head outputs the trajectory.

`Model B` replaces the custom visual encoder with a pretrained `torchvision` ResNet18 backbone used as a camera feature extractor. It keeps the history branch simple, uses a compact fusion MLP, and still predicts the same trajectory format.

`Model B V2` keeps that same overall architecture, but is the recommended version because it has clearer pretrained input handling and a safer fine-tuning setup for the ResNet18 backbone.

All models return a predicted future trajectory with shape `(batch_size, 60, 3)`. Because every variant preserves the same `forward(camera, history)` interface and output format, comparisons between models stay straightforward.

Use the notebook parameter:

```python
MODEL_NAME = "model_b_v2"   # current recommended; alternatives: "baseline", "model_a", or "model_b"
```

Internally, model creation goes through:

```python
model = build_model(MODEL_NAME)
```

This keeps comparisons simple and avoids overwriting old architectures.

For the ResNet18 models, the optimizer helper supports using a smaller learning rate on the pretrained backbone than on the newly added head layers:

```python
optimizer = build_optimizer(
    model,
    learning_rate=3e-4,
    weight_decay=1e-4,
    backbone_lr_scale=0.1,
)
```

`Model B V2` keeps the same optimizer call, but defaults to a `0.1x` backbone learning-rate scale when no explicit backbone LR override is provided.

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
    early_stopping_patience=25,
    early_stopping_min_delta=1e-3,
)
```

A short frozen-backbone warmup is also supported for the ResNet18 variants. During this warmup, the head learns first while the pretrained backbone stays frozen; after that, the backbone is unfrozen for normal fine-tuning.

```python
BACKBONE_WARMUP_EPOCHS = 2
```

## Training, Checkpoints, And Outputs

Training is launched from the notebook. The notebook builds the datasets and dataloaders, creates the model through `build_model(...)`, builds the optimizer and scheduler, and then calls `train(...)` from `src/train.py`.

During training, the loop:

- runs training on the train split,
- evaluates on the validation split every epoch,
- tracks validation loss, ADE, and FDE,
- saves the best checkpoint by validation ADE,
- optionally applies early stopping based on validation ADE,
- optionally freezes the ResNet18 backbone for the first `BACKBONE_WARMUP_EPOCHS` epochs before unfreezing it.

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

The training setup tracks the best validation ADE during training.

- The best checkpoint is saved to `model.pth`.
- The last model is saved separately to `model_last.pth`.
- By default, the notebook reloads the best checkpoint before qualitative evaluation and submission generation.
- Early stopping, when enabled, only decides when to stop training; it does not change the fact that `model.pth` is always selected by best validation ADE.

This behavior is controlled by:

```python
RELOAD_BEST_CHECKPOINT_FOR_INFERENCE = True
```

So, by default, the submission CSV is generated from the best checkpoint of the run, not just the last epoch.

## Inference And Submission Generation

Inference and submission generation are launched from the notebook and implemented through helpers in `src/submission.py`.

In this repository, "inference" means running the trained planner forward without labels. The most important inference pass is on `data/test_public/`, where the model produces the final submission CSV. The notebook also runs downstream qualitative inspection on validation examples after the best checkpoint is optionally reloaded.

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
