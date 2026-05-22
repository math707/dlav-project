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

## Best Observed Validation Result / Recommended Default

Best observed validation ADE: `1.7429`

Best known Kaggle public score from the runs documented locally: approximately `1.56` from
`20260515_090908_phase3_resnet18_realmix500_pretrained_aug0.6_plateau_wd0.0001`.
The later `520`-real run did not improve the public score (`1.57` reported) despite using more real-data mixing.

Recommended notebook defaults:

- `MODEL_NAME = 'phase3_resnet18'`
- `BATCH_SIZE = 32`
- `NUM_EPOCHS = 120`
- `LR = 7e-4`
- `WEIGHT_DECAY = 1e-4`
- `REAL_TRAIN_COUNT = 500`
- `USE_AUGMENTATION = True`
- `AUGMENTATION_STRENGTH = 0.6`
- `PRETRAINED = True`
- `BACKBONE_LR_SCALE = 0.1`
- `BACKBONE_WARMUP_EPOCHS = 3`
- `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE = True`

Recommended two-stage experiment defaults:

- `USE_TWO_STAGE_TRAINING = True`
- `REAL_TRAIN_COUNT_STAGE1 = 700`
- Stage 1: `STAGE1_LR = 5e-4`, `STAGE1_AUGMENTATION_STRENGTH = 0.6`, `STAGE1_NUM_EPOCHS = 100`, `STAGE1_EARLY_STOPPING_PATIENCE = 18`
- Stage 2: `STAGE2_LR = 1e-4`, `STAGE2_AUGMENTATION_STRENGTH = 0.0`, `STAGE2_NUM_EPOCHS = 20`, `STAGE2_EARLY_STOPPING_PATIENCE = 8`
- `TOP_K_CHECKPOINTS = 5`
- `GENERATE_LAST_CHECKPOINT_SUBMISSION = True`
- `GENERATE_TOP_K_SUBMISSIONS = True`
- `GENERATE_ENSEMBLE_SUBMISSION = True`
- `ENSEMBLE_CHECKPOINT_LIMIT = 3`

This setup is preferred because the pretrained ResNet18 backbone gives strong image features from the start, mixing labeled real samples directly targets the sim-to-real gap, and photometric augmentation improves robustness without changing trajectory geometry. Restricting the objective to XY also keeps training, validation, and submission perfectly aligned.

## How to Train

1. Open [DLAV_Phase3.ipynb](DLAV_Phase3.ipynb).
2. Edit the top configuration cell. The main knobs are `REAL_TRAIN_COUNT`, `USE_AUGMENTATION`, `AUGMENTATION_STRENGTH`, `PRETRAINED`, `BATCH_SIZE`, `NUM_EPOCHS`, and `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE`.
3. Run the notebook from top to bottom.

The notebook handles setup, dataset download, randomized real-data splitting, one-stage or two-stage training, validation, checkpoint selection by `val_ADE`, checkpoint reload, and submission generation.

Optional experiment infrastructure:

- `TOP_K_CHECKPOINTS` keeps the best `k` validation checkpoints by `val_ADE` in the run folder.
- `GENERATE_LAST_CHECKPOINT_SUBMISSION` writes a separate CSV for the last checkpoint.
- `GENERATE_TOP_K_SUBMISSIONS` writes separate CSVs for the saved top-k checkpoints.
- `GENERATE_ENSEMBLE_SUBMISSION` averages multiple checkpoint predictions into one extra CSV.
- `ENSEMBLE_CHECKPOINT_LIMIT` chooses how many of the saved top-k checkpoints are averaged for the ensemble CSV.

## Two-Stage Training

- Stage 1 trains on the mixed synthetic + real split.
- Stage 2 reloads the best Stage 1 checkpoint and fine-tunes on the same `real_train` split only.
- Validation stays on the held-out `real_val` split for both stages.
- Inference still uses only `camera` and `sdc_history_feature`.

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

- The notebook keeps `EXPECTED_PUBLIC_TEST_SAMPLES = None` and infers the public test size automatically.
