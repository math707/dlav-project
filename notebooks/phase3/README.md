# Phase 3

Notebook: [DLAV_Phase3.ipynb](DLAV_Phase3.ipynb)  
Code: `src/phase3/`

## Goal

Generalize from simulation to real data. Training mixes synthetic `train/` samples with a randomized subset of labeled real `val_real/` samples, and validation is done on the remaining real samples.

## Inputs and Targets

- Training and validation inputs: `camera`, `sdc_history_feature`
- Target source: `sdc_future_feature`
- Optimized target in this repo: XY coordinates only, shape `[60, 2]`
- Inference inputs: `camera`, `sdc_history_feature`
- Not used in Phase 3: `depth`, `semantic_label`, `driving_command`
- Submission output: 60 future XY coordinates

## Implemented Model

| Model | Notes |
| --- | --- |
| `phase3_resnet18` | Pretrained ResNet18 camera backbone, history MLP, fusion MLP, XY trajectory head |

## Current Best Kaggle Setup / Recommended Default

Current recommended mode: `TRAINING_MODE = 'legacy_one_stage'`

Best known Kaggle public score: approximately `1.35`

Main parameters to display in the notebook config:

- `MODEL_NAME = 'phase3_resnet18'`
- `TRAINING_MODE = 'legacy_one_stage'`
- `BATCH_SIZE = 32`
- `TEST_BATCH_SIZE = 250`
- `NUM_EPOCHS = 120`
- `LR = 7e-4`
- `WEIGHT_DECAY = 1e-4`
- `REAL_TRAIN_COUNT = 800`
- `USE_AUGMENTATION = True`
- `AUGMENTATION_STRENGTH = 0.65`
- `PRETRAINED = True`
- `BACKBONE_LR_SCALE = 0.1`
- `BACKBONE_WARMUP_EPOCHS = 3`
- `EARLY_STOPPING_PATIENCE = 20`
- `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE = True`
- `EXPECTED_PUBLIC_TEST_SAMPLES = None`

`legacy_one_stage` reproduces the older one-stage behavior and is currently the best setup for Kaggle. `one_stage` and `two_stage` remain available for experimentation, but `two_stage` is not the recommended setup at the moment because it underperformed in our tests. The public test size is inferred automatically from `len(public_test_files)` and is currently `864`.

## How to Train

1. Open [DLAV_Phase3.ipynb](DLAV_Phase3.ipynb).
2. Edit the top configuration cell. For the current best setup, start with `TRAINING_MODE = 'legacy_one_stage'`, then adjust `REAL_TRAIN_COUNT`, `USE_AUGMENTATION`, `AUGMENTATION_STRENGTH`, `PRETRAINED`, `BATCH_SIZE`, `NUM_EPOCHS`, and `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE`.
3. Run the notebook from top to bottom.

The notebook handles setup, dataset download, randomized real-data splitting, one-stage or two-stage training, validation, checkpoint selection by `val_ADE`, checkpoint reload, and submission generation.

## Training Modes

- `TRAINING_MODE = 'legacy_one_stage'` reproduces the pre-two-stage notebook flow as closely as possible. It uses the direct one-stage `train(...)` call, keeps the original mixed-train / held-out-real validation behavior, saves the best and last checkpoints, and only generates the primary submission CSV by default.
- `TRAINING_MODE = 'one_stage'` keeps one-stage training on the mixed split but uses the newer notebook plumbing, including optional top-k checkpoint saving and extra submission variants.
- `TRAINING_MODE = 'two_stage'` runs mixed Stage 1 and real-only Stage 2 fine-tuning. It is currently experimental rather than the recommended best setup.

Optional experiment infrastructure:

- `TOP_K_CHECKPOINTS` keeps the best `k` validation checkpoints by `val_ADE` in the run folder.
- `GENERATE_LAST_CHECKPOINT_SUBMISSION` writes a separate CSV for the last checkpoint.
- `GENERATE_TOP_K_SUBMISSIONS` writes separate CSVs for the saved top-k checkpoints.
- `GENERATE_ENSEMBLE_SUBMISSION` averages multiple checkpoint predictions into one extra CSV.
- `ENSEMBLE_CHECKPOINT_LIMIT` chooses how many of the saved top-k checkpoints are averaged for the ensemble CSV.

In `legacy_one_stage` mode, the notebook deliberately falls back to the old behavior: top-k checkpointing is reduced to `1`, and the extra last/top-k/ensemble submission variants are skipped.

## Two-Stage Training

- Stage 1 trains on the mixed synthetic + real split.
- Stage 2 reloads the best Stage 1 checkpoint and fine-tunes on the same `real_train` split only.
- Validation stays on the held-out `real_val` split for both stages.
- Inference still uses only `camera` and `sdc_history_feature`.
- This mode is kept for experimentation and is not the current recommended best setup.

## How to Reload / Infer / Submit

1. Keep `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE = True` to reload the best checkpoint selected by validation ADE.
2. Run the reload/validation cell.
3. Run the submission cell to generate `submission_phase3.csv`.

The submission contains `id, x_1, y_1, ..., x_60, y_60`. Only XY trajectory coordinates are optimized and submitted.

## Outputs

- Run folder: `outputs/runs/phase3/<timestamp>_<run_name>/`
- Best checkpoint: `model.pth`
- Last checkpoint: `model_last.pth`
- Run submission: `outputs/runs/phase3/<timestamp>_<run_name>/submission_phase3.csv`
- Legacy checkpoint copy: `outputs/checkpoints/phase3/phase3_model.pth`
- Legacy submission copy: `outputs/submissions/phase3/submission_phase3.csv`

## Notes

- The notebook keeps `EXPECTED_PUBLIC_TEST_SAMPLES = None` and infers the public test size automatically. The current public test set contains `864` samples.
