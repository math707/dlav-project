# Phase 3

## Goal

Phase 3 focuses on sim-to-real generalization. The model is trained on synthetic `train/` data and mixed with a randomized subset of labeled real `val_real/` samples, then validated on the remaining real samples.

## Inputs and Targets

- Training and validation inputs:
  - `camera`
  - `sdc_history_feature`
- Training and validation target:
  - `sdc_future_feature`
- Target actually optimized in this repo:
  - XY coordinates only, shape `[60, 2]`
- Test-time inputs:
  - `camera`
  - `sdc_history_feature`
- Inputs intentionally not used in Phase 3:
  - `depth`
  - `semantic_label`
  - `driving_command`
- Submission output:
  - predicted XY coordinates for 60 future steps

## Repo Structure for This Phase

- Notebook entry point: `notebooks/phase3/DLAV_Phase3.ipynb`
- Starter reference notebook: `notebooks/starter/DLAV_Phase3_starter_reference.ipynb`
- Phase code: `src/phase3/`
  - `dataset.py`
  - `augmentations.py`
  - `model.py`
  - `train.py`
  - `submission.py`
- Shared infrastructure:
  - `src/shared/project_setup.py`
  - `src/shared/run_utils.py`
  - `src/shared/training_setup.py`
  - `src/shared/data_utils.py`

## Implemented Model

- `phase3_resnet18`: pretrained ResNet18 camera backbone, MLP history encoder, fusion MLP, and a `[60, 2]` trajectory head

## Recommended Default Config

- `MODEL_NAME = 'phase3_resnet18'`
- `NUM_EPOCHS = 100`
- `BATCH_SIZE = 32`
- `LR = 1e-3`
- `WEIGHT_DECAY = 1e-4`
- `REAL_TRAIN_COUNT = 500`
- `USE_AUGMENTATION = True`
- `PRETRAINED = True`
- `BACKBONE_LR_SCALE = 0.1`
- `BACKBONE_WARMUP_EPOCHS = 2`
- `SCHEDULER_PATIENCE = 6`
- `EARLY_STOPPING_PATIENCE = 25`

## Training Instructions

1. Open [DLAV_Phase3.ipynb](DLAV_Phase3.ipynb).
2. Edit the configuration cell at the top of the notebook.
3. The main variables to check are:
   - `REAL_TRAIN_COUNT`
   - `USE_AUGMENTATION`
   - `AUGMENTATION_STRENGTH`
   - `PRETRAINED`
   - `BATCH_SIZE`
   - `NUM_EPOCHS`
   - `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE`
4. Run the notebook from top to bottom.

The notebook handles environment setup, dataset download, randomized real-data splitting, training, validation, checkpoint selection by `val_ADE`, checkpoint reload, and submission generation.

## Inference and Submission Instructions

1. Use the same notebook after training.
2. Keep `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE = True` if you want inference and submission to use the checkpoint selected by validation ADE.
3. Run the reload/validation cell and then the submission cell.

Important Phase 3 behavior:

- Only XY targets are optimized and evaluated.
- Validation loss is computed on XY only, matching training.
- The Kaggle CSV contains `id, x_1, y_1, ..., x_60, y_60`.

## Output Locations

- Run directory: `outputs/runs/phase3/<timestamp>_<run_name>/`
- Best checkpoint inside the run: `model.pth`
- Last-epoch checkpoint inside the run: `model_last.pth`
- Legacy checkpoint copy: `outputs/checkpoints/phase3/phase3_model.pth`
- Run-scoped submission: `outputs/runs/phase3/<timestamp>_<run_name>/submission_phase3.csv`
- Legacy submission copy: `outputs/submissions/phase3/submission_phase3.csv`

## Notes / Assumptions

- The implementation expects `camera` to be stored as `[H, W, 3]`.
- The implementation expects `sdc_history_feature` to have shape `[21, 3]`.
- The implementation expects `sdc_future_feature` to have shape `[60, 3]` or `[60, >=2]`, and keeps only XY.
- Augmentations are photometric only by default so future labels stay geometrically consistent.
