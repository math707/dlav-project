# Phase 2

## Goal

Phase 2 focuses on perception-aware planning. The main task is still future trajectory prediction, evaluated through trajectory quality metrics such as ADE, while adding richer perception signals during training.

## Inputs and Targets

- Training and validation inputs:
  - `camera`
  - `sdc_history_feature`
  - `driving_command`
- Main training and validation target:
  - `sdc_future_feature`
- Optional auxiliary training target:
  - `depth`
- Test-time inputs:
  - `camera`
  - `sdc_history_feature`
  - `driving_command`
- Test-time inputs that are not required:
  - `depth`
  - `semantic_label`
  - `sdc_future_feature`
- Submission output:
  - predicted XY coordinates for 60 future steps

## Repo Structure for This Phase

- Notebook entry point: `notebooks/phase2/DLAV_Phase2.ipynb`
- Starter reference notebook: `notebooks/starter/DLAV_Phase2_starter_reference.ipynb`
- Phase code: `src/phase2/`
  - `dataset.py`
  - `model.py`
  - `train.py`
  - `submission.py`
- Shared infrastructure:
  - `src/shared/project_setup.py`
  - `src/shared/run_utils.py`
  - `src/shared/training_setup.py`
  - `src/shared/data_utils.py`

## Implemented Models

- `phase2_trajectory_only`: trajectory baseline using camera, motion history, and `driving_command`
- `phase2_multitask`: shared planner plus a depth estimation auxiliary head used during training and validation

## Recommended Model/Config

Current main candidate under evaluation:

- `MODEL_NAME = 'phase2_multitask'`
- `USE_DEPTH_AUX = True`

Current notebook defaults also use:

- `DEPTH_LOSS_NAME = 'l1'`
- `LAMBDA_DEPTH = 0.05`
- `BACKBONE_WARMUP_EPOCHS = 2`

Why this is the current recommendation:

- Depth auxiliary supervision can encourage the shared visual backbone to learn scene geometry, obstacle distance, and spatial layout that should help planning.
- The depth branch uses the shared visual features directly, without detaching them, so the auxiliary loss can influence the encoder instead of only training a separate head.
- This keeps the main task unchanged at inference time: trajectory prediction still uses camera, history, and command only.

Status of final model selection:

- `phase2_trajectory_only` is kept as the reference baseline.
- `phase2_multitask` with depth supervision is the current main candidate under evaluation.
- The final Phase 2 choice will be based on validation ADE comparisons between the trajectory-only baseline and multitask depth variants with different `LAMBDA_DEPTH` values.
- Final selected run and final ADE/Kaggle result: to be filled after the final Phase 2 experiments

## Training Instructions

1. Open [DLAV_Phase2.ipynb](DLAV_Phase2.ipynb).
2. Edit the configuration cell at the top of the notebook.
3. The main variables to check are:
   - `MODEL_NAME`
   - `USE_DEPTH_AUX`
   - `LAMBDA_DEPTH`
   - `DEPTH_LOSS_NAME`
   - `BATCH_SIZE`
   - `NUM_EPOCHS`
   - `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE`
4. Run the notebook from top to bottom.

The notebook handles environment setup, dataset loading, training, validation, best-checkpoint selection by `val_ADE`, checkpoint reload, and submission generation.

## Inference and Submission Instructions

1. Use the same notebook after or between training runs.
2. Keep `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE = True` if you want the submission to use the checkpoint selected by validation ADE.
3. Run the reload/validation section and then the submission section.

Important Phase 2 behavior:

- Depth is training supervision only.
- Depth is not an inference input.
- Depth is not a submission output.

## Output Locations

- Run directory: `outputs/runs/phase2/<timestamp>_<run_name>/`
- Best checkpoint inside the run: `model.pth`
- Last-epoch checkpoint inside the run: `model_last.pth`
- Legacy checkpoint copy: `outputs/checkpoints/phase2/phase2_model.pth`
- Run-scoped submission: `outputs/runs/phase2/<timestamp>_<run_name>/submission_phase2.csv`
- Legacy submission copy: `outputs/submissions/phase2/submission_phase2.csv`

## Notes / Limitations / Next Steps

- Semantic segmentation is intentionally not implemented yet in this first clean Phase 2 pass.
- A future extension is to add semantic supervision on top of the current trajectory and depth setup.
- This README does not claim final Phase 2 metrics because the validation runs are still in progress.
