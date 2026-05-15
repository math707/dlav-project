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

This setup is preferred because the pretrained ResNet18 backbone gives strong image features from the start, mixing labeled real samples directly targets the sim-to-real gap, and photometric augmentation improves robustness without changing trajectory geometry. Restricting the objective to XY also keeps training, validation, and submission perfectly aligned.

## How to Train

1. Open [DLAV_Phase3.ipynb](DLAV_Phase3.ipynb).
2. Edit the top configuration cell. The main knobs are `REAL_TRAIN_COUNT`, `USE_AUGMENTATION`, `AUGMENTATION_STRENGTH`, `PRETRAINED`, `BATCH_SIZE`, `NUM_EPOCHS`, and `RELOAD_BEST_CHECKPOINT_FOR_INFERENCE`.
3. Run the notebook from top to bottom.

The notebook handles setup, dataset download, randomized real-data splitting, training, validation, checkpoint selection by `val_ADE`, checkpoint reload, and submission generation.

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
