# Phase 2

Notebook: [DLAV_Phase2.ipynb](DLAV_Phase2.ipynb)  
Code: `src/phase2/`

## Goal

Predict the future trajectory while using richer scene understanding during training. The main output is still the future XY trajectory, and depth is used only as auxiliary supervision.

## Inputs and Targets

- Training and validation inputs: `camera`, `sdc_history_feature`, `driving_command`
- Main target: `sdc_future_feature`
- Auxiliary training target: `depth`
- Inference inputs: `camera`, `sdc_history_feature`, `driving_command`
- Not used at inference: `depth`, `semantic_label`
- Submission output: 60 future XY coordinates

## Implemented Models

| Model | Notes |
| --- | --- |
| `phase2_trajectory_only` | Command-conditioned trajectory baseline |
| `phase2_multitask` | Baseline planner with depth auxiliary head |
| `phase2_b_v2_port` | Phase 1 `model_b_v2`-style ablation |
| `phase2_b_v2_depth` | `model_b_v2`-style ablation with depth supervision |
| `phase2_gru_delta_residual` | GRU-based residual trajectory decoder |
| `phase2_temporal_delta_depth` | Temporal residual decoder with depth auxiliary supervision |

## Best Observed Validation Result / Recommended Default

Best observed validation ADE: `1.5422`

Recommended notebook defaults:

- `MODEL_NAME = 'phase2_temporal_delta_depth'`
- `USE_DEPTH_AUX = True`
- `LAMBDA_DEPTH = 0.05`
- `DEPTH_LOSS_NAME = 'l1'`
- `BATCH_SIZE = 32`
- `NUM_EPOCHS = 160`
- `LR = 1e-3`
- `WEIGHT_DECAY = 1e-4`
- `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE = True`

This configuration is preferred because the temporal decoder gives the trajectory head an explicit time structure instead of regressing all future points with one flat head. The auxiliary depth loss also helps the visual backbone learn scene geometry, while keeping inference unchanged: depth is used for training only.

## How to Train

1. Open [DLAV_Phase2.ipynb](DLAV_Phase2.ipynb).
2. Edit the top configuration cell. The main knobs are `MODEL_NAME`, `USE_DEPTH_AUX`, `LAMBDA_DEPTH`, `DEPTH_LOSS_NAME`, `BATCH_SIZE`, `NUM_EPOCHS`, and `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE`.
3. Run the notebook from top to bottom.

The notebook handles setup, dataset loading, training, validation, checkpoint selection by `val_ADE`, checkpoint reload, and submission generation.

## How to Reload / Infer / Submit

1. Keep `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE = True` to reload the best checkpoint selected by validation ADE.
2. Run the reload/validation section.
3. Run the submission section to generate `submission_phase2.csv`.

Depth is not an inference input and is not part of the submission output.

## Outputs

- Run folder: `outputs/runs/phase2/<timestamp>_<run_name>/`
- Best checkpoint: `model.pth`
- Last checkpoint: `model_last.pth`
- Run submission: `outputs/runs/phase2/<timestamp>_<run_name>/submission_phase2.csv`
- Legacy checkpoint copy: `outputs/checkpoints/phase2/phase2_model.pth`
- Legacy submission copy: `outputs/submissions/phase2/submission_phase2.csv`

## Notes

- Depth is auxiliary supervision only: it is not required at inference time and is not part of the submission.
