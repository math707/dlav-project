# Phase 1

Notebook: [DLAV_Phase1.ipynb](DLAV_Phase1.ipynb)  
Code: `src/phase1/`

## Goal

Predict the future trajectory of the self-driving car from the front camera and motion history. The official Phase 1 task also allows `driving_command`, but the recommended implementation keeps the input pipeline simple and uses `camera` and `sdc_history_feature`; explicit command conditioning is introduced in Phase 2.

## Inputs and Targets

- Official allowed inputs: `camera`, `sdc_history_feature`, `driving_command`
- Inputs used by the recommended model: `camera`, `sdc_history_feature`
- Target: `sdc_future_feature`
- Submission output: 60 future XY coordinates

## Implemented Models

| Model | Notes |
| --- | --- |
| `baseline` | Small camera encoder plus direct trajectory head |
| `model_a` | Stronger custom CNN with a history encoder |
| `model_b` | ResNet18-based planner |
| `model_b_v2` | Cleaned-up ResNet18-based variant and current default |

## Best Observed Validation Result / Recommended Default

Best observed validation ADE: `1.6783`

Recommended notebook defaults:

- `MODEL_NAME = 'model_b_v2'`
- `BATCH_SIZE = 32`
- `NUM_EPOCHS = 100`
- `LEARNING_RATE = 7e-4`
- `WEIGHT_DECAY = 1e-4`
- `BACKBONE_WARMUP_EPOCHS = 2`
- `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE = True`

`model_b_v2` is the preferred Phase 1 model because the pretrained ResNet18 backbone extracts stronger visual features than the simpler CNN baselines, while the rest of the planner stays simple and stable. In practice, it gives the best validation performance in this repo without making inference more complex.

## How to Train

1. Open [DLAV_Phase1.ipynb](DLAV_Phase1.ipynb).
2. Edit the top configuration cell. The main knobs are `MODEL_NAME`, `LEARNING_RATE_NAME`, `BATCH_SIZE`, `NUM_EPOCHS`, and `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE`.
3. Run the notebook from top to bottom.

The notebook handles setup, data resolution, training, validation, checkpoint selection, and submission generation.

## How to Reload / Infer / Submit

1. Keep `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE = True` to reload the best checkpoint selected by validation ADE.
2. Run the validation reload cell.
3. Run the submission cell to generate `submission_phase1.csv`.

## Outputs

- Run folder: `outputs/runs/phase1/<timestamp>_<run_name>/`
- Best checkpoint: `model.pth`
- Last checkpoint: `model_last.pth`
- Run submission: `outputs/runs/phase1/<timestamp>_<run_name>/submission_phase1.csv`
- Legacy checkpoint copy: `outputs/checkpoints/phase1/phase1_model.pth`
- Legacy submission copy: `outputs/submissions/phase1/submission_phase1.csv`

## Notes

- `BACKBONE_WARMUP_EPOCHS = 2` is kept as the current safe notebook default.
