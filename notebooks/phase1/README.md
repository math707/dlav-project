# Phase 1

## Goal

Phase 1 predicts the future trajectory of the self-driving car. In the official project setting, the allowed Phase 1 inputs include the front camera, motion history, and driving command. The selected Phase 1 implementation in this repo mainly uses camera and motion history, while explicit command conditioning is introduced in Phase 2.

## Inputs and Targets

- Official allowed Phase 1 inputs:
  - `camera`
  - `sdc_history_feature`
  - `driving_command`
- Inputs used by the selected Phase 1 implementation:
  - `camera`
  - `sdc_history_feature`
- Training and validation target:
  - `sdc_future_feature`
- Test-time inputs used by the selected Phase 1 implementation:
  - `camera`
  - `sdc_history_feature`
- Submission output:
  - predicted XY coordinates for 60 future steps

## Repo Structure for This Phase

- Notebook entry point: `notebooks/phase1/DLAV_Phase1.ipynb`
- Phase code: `src/phase1/`
  - `dataset.py`
  - `model.py`
  - `train.py`
- Shared infrastructure:
  - `src/shared/project_setup.py`
  - `src/shared/run_utils.py`
  - `src/shared/training_setup.py`
  - `src/shared/submission.py`

## Implemented Models

- `baseline`: simple camera encoder plus flattened history and a direct trajectory decoder
- `model_a`: stronger custom CNN with a dedicated history encoder and fusion MLP
- `model_b`: ResNet18-style pretrained visual backbone with learned projection and stronger fusion
- `model_b_v2`: refined ResNet18-based variant with cleaner pretrained-backbone handling and backbone-specific optimization defaults

## Recommended Model/Config

Current recommendation:

- `MODEL_NAME = 'model_b_v2'`

Why this model is currently preferred:

- It is the strongest Phase 1 direction tested so far in the validation-oriented workflow used in this repo.
- It keeps the inference path clean: one planner, one best-checkpoint selection rule, one submission path.
- It builds on the strongest visual backbone used in Phase 1 while keeping the notebook setup stable.

Final selected metrics:

- Best validation ADE from the selected run: to be filled after the final run

## Training Instructions

1. Open [DLAV_Phase1.ipynb](DLAV_Phase1.ipynb).
2. Edit the configuration cell at the top of the notebook.
3. The main variables to check are:
   - `MODEL_NAME`
   - `BATCH_SIZE`
   - `NUM_EPOCHS`
   - `LEARNING_RATE_NAME`
   - `USE_LR_SCHEDULER`
   - `EARLY_STOPPING_PATIENCE`
   - `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE`
4. Run the notebook from top to bottom.

The notebook handles environment setup, dataset resolution, training, validation, checkpoint selection, and submission generation.

## Inference and Submission Instructions

1. Use the same notebook after training.
2. Keep `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE = True` if you want inference and submission to use the best checkpoint selected by validation ADE.
3. Run the validation reload cell and then the submission cell.

The Phase 1 submission flow uses `src/shared/submission.py`.

## Output Locations

- Run directory: `outputs/runs/phase1/<timestamp>_<run_name>/`
- Best checkpoint inside the run: `model.pth`
- Last-epoch checkpoint inside the run: `model_last.pth`
- Legacy checkpoint copy: `outputs/checkpoints/phase1/phase1_model.pth`
- Run-scoped submission: `outputs/runs/phase1/<timestamp>_<run_name>/submission_phase1.csv`
- Legacy submission copy: `outputs/submissions/phase1/submission_phase1.csv`

## Notes / Limitations / Next Steps

- This README does not claim final ADE numbers because the final selected run may still change.
- The root-level `notebooks/DLAV_Phase1.ipynb` file remains only as a temporary compatibility copy.
