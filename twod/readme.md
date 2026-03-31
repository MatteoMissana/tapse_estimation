# 2D pipeline (tapse_estimation/2d)

This README explains the purpose and layout of the `2d/` pipeline, what each folder/file does and how to use the main scripts:
- [`2d/training.py`](2d/training.py) — training pipeline
- [`2d/indices_prediction.py`](2d/indices_prediction.py) — inference + index calculation pipeline
- [`2d/statystical_analysis.py`](2d/statystical_analysis.py) — statistical / Bland–Altman analysis from the excel file produced by indices_prediction
- [`2d/test_models.py`](2d/test_models.py) — per-h5 testing & diagnostics

Use the links below to jump straight to the files in the workspace:
- [2d/training.py](2d/training.py)
- [2d/indices_prediction.py](2d/indices_prediction.py)
- [2d/statystical_analysis.py](2d/statystical_analysis.py)
- [2d/test_models.py](2d/test_models.py)

Other useful files & modules (quick links):
- Data I/O and preprocessing: [2d/dataloader/main.py](2d/dataloader/main.py), [2d/dataloader/preprocessing.py](2d/dataloader/preprocessing.py)
- Models & init: [2d/models/models.py](2d/models/models.py), [2d/models/weights_initialization.py](2d/models/weights_initialization.py)
- Losses: [2d/losses/distances.py](2d/losses/distances.py)
- Postprocessing & metric helpers: [2d/postprocessing/coordinates_calculation_from_masks.py](2d/postprocessing/coordinates_calculation_from_masks.py), [2d/postprocessing/indices_calculation.py](2d/postprocessing/indices_calculation.py), [2d/postprocessing/kalman_filter.py](2d/postprocessing/kalman_filter.py), [2d/postprocessing/pan_tompkins.py](2d/postprocessing/pan_tompkins.py)
- Callbacks: [2d/callbacks/early_stopping.py](2d/callbacks/early_stopping.py), [2d/callbacks/lr_schedule.py](2d/callbacks/lr_schedule.py)
- Example augmentations: [2d/augmentations/aug0.py](2d/augmentations/aug0.py)
- Model checkpoints: [2d/model_weights/](2d/model_weights/)

---

## Overview

This folder implements the 2D approach for predicting three anatomical landmarks per frame (two annular points + apex) and deriving cardiac indices (TAPSE, RVFAC, RVLSF, areas, distances). Pipeline stages:

1. Training: train a 2D model that predicts per-frame segmentation or heatmaps (see [2d/training.py](2d/training.py)).
2. Inference & index calculation: run the trained model on HDF5 sequences, extract keypoints via center-of-mass and compute indices per heartbeat (see [`indices_prediction.predict_indices`](2d/indices_prediction.py)).
3. Per-file testing / visualization: generate diagnostic images and per-file stats (see [`test_models.process_h5_file_single`](2d/test_models.py)).
4. Statistical evaluation: merge manual and automatic results, compute Bland–Altman stats, correlation and save Excel/plots (see [`statystical_analysis.analysis`](2d/statystical_analysis.py)).

---

## How to use the main scripts

Notes:
- All scripts are executable as CLI Python scripts (they expose a `main()` with `argparse`).
- Typical device selection is automatic (CUDA if available). See each script header for arguments (open the linked file to read the full arg list).

### 1) Training: [2d/training.py](2d/training.py)

What it does:
- Loads datasets using `KeypointDataset` ([2d/dataloader/main.py](2d/dataloader/main.py)).
- Builds model (many options supported: monai U-Net, ResNet regressors, etc.) from [2d/models/models.py](2d/models/models.py).
- Trains using OrderedDistanceLoss or GaussianKeypointLoss ([2d/losses/distances.py](2d/losses/distances.py)). (mainly ordered distance was used)
- Supports wandb logging, checkpointing, lr-scheduling, early stopping.

Typical example (adjust paths/args to your environment):
python 2d/training.py \
  --train_data "D:\mmissana\data\RV_PATIENTS\dataset_after_review\train.npz" \  (the dataset needs to be formatted in 3 npz files, each file containing file['images'] and file['keypoints']. Keypoints are ordered as FW annular point, septal ann. point and apex.)
  --val_data "D:\mmissana\data\RV_PATIENTS\dataset_after_review\val.npz" \
  --epochs 300 --batch_size 16 \
  --model "monai_U-Net" --start_filts 8 --depth 6 \
  --loss ordered_distance \
  --checkpoint_path "checkpoints" \ I trained all models from scratch (--from_scratch argument). This is the path where to save the weights.
  --save_model_path "2d/runs/my_experiment"

Hints:
- If resuming from a checkpoint, pass `--model_path <path_to_checkpoint>` and ensure the architecture args match.
- Set random seed args for reproducibility (`--seed`).

See: [`2d/training.py`](2d/training.py) and [2d/models/models.py](2d/models/models.py).

---

### 2) Inference & indices: [2d/indices_prediction.py](2d/indices_prediction.py)

What it does:
- Reads an HDF5 file with tissue frames, tissue times, ECG and pixel size.
- Runs the trained U-Net (`models.Unet`) to get output channels per keypoint, extracts centers via [`postprocessing.coordinates_calculation_from_masks.center_of_mass`](2d/postprocessing/coordinates_calculation_from_masks.py).
- Optionally smooths coordinates with Kalman filter ([2d/postprocessing/kalman_filter.py](2d/postprocessing/kalman_filter.py)).
- Detects r-peaks with Pan-Tompkins ([2d/postprocessing/pan_tompkins.py](2d/postprocessing/pan_tompkins.py)) to segment heartbeats.
- Computes per-beat indices with [`postprocessing.indices_calculation.tric_apex_distance_calculation`](2d/postprocessing/indices_calculation.py), `tapse_calculation`, area metrics and reductions (mean/max/min or best combination).
- Writes Excel rows into the provided metadata Excel (path passed to script).

Common CLI options:
- --h5_dir DIR (directory with .h5 files)
- --excel_path PATH (excel containing patient metadata / paths)
- --model_path FILE (path to saved model checkpoint)
- --filters, --depth, --residuals (model architecture params)
- --filter (apply Kalman filter) sorry for the similar name
- --tapse {distance,projection} tells how to calculate tapse (see my thesis)
- --reduction {mean,max,min} how to ''average'' indexes
- --save_images (save visualizations)
- --threshold (center-of-mass threshold) (0.875 for the unet)
- --two_dimensional (flag for reorientation) Because in theory it works also for the reslices extracted from the 3d data, but could be simplified (depends on the kind of images youre using)
- --best_combination (use heuristic combination of min/max/mean across indices) Not sure if it's actuallly the best for each index

Example:
d:/mmissana/tapse_estimation/2d/indices_prediction.py --h5_dir D:\mmissana\data\RV_PATIENTS\RV_patients_converted --model_path D:\mmissana\tapse_estimation\2d\runs\best_unet\best_model.pth --depth 6 --filters 16 --residuals 0 --tapse distance --reduction max --excel_path D:\mmissana\tapse_estimation\2d\results\best_combination_best_rvfac\best_unet.xlsx --threshold 0.875 --two_dimensional --area_method spline --best_combination

See: [`2d/indices_prediction.py`](2d/indices_prediction.py), [`2d/postprocessing/indices_calculation.py`](2d/postprocessing/indices_calculation.py), [`2d/postprocessing/kalman_filter.py`](2d/postprocessing/kalman_filter.py).

---

### 3) Statistical analysis: [2d/statystical_analysis.py](2d/statystical_analysis.py)

What it does:
- Loads manual / automatic Excel files and a fixed set of patient IDs (default is the curated list).
- Merges annotation and prediction columns (suffix `_ann` and `_pred`), computes:
  - Per-index mean error, variance, std, 95% LoA (Bland–Altman)
  - Pearson and Spearman correlations
  - Mean % error
- Saves per-patient differences and a `*.xlsx` file with the summarized stats and Bland–Altman plots.

Typical usage:
python 2d/statystical_analysis.py --automatic_path "2d/results/XXX/best_unet.xlsx"

Remarks:
- The script expects a manual reference Excel coded in the script by default (see top of file); change `manual_path` in the CLI wrapper or the file if needed.
- If `best_combination` or `best_rvlsffw` names are present in automatic Excel, script derives RVFAC or RV strain fw in a different way to minimize bias (see my thesis).

See: [`2d/statystical_analysis.py`](2d/statystical_analysis.py).

---

### 4) Per-file testing & visualization: [2d/test_models.py](2d/test_models.py)
Key symbol:
- [`test_models.process_h5_file_single`](2d/test_models.py)

What it does:
- Walks through annotated folders or specified datasets, runs inference frame-by-frame (no batching), extracts coordinates, computes distances to annotations and optional per-frame diagnostics.
- Saves:
  - visualizations with predicted and annotated keypoints ([utils.plot.save_image_ann_pred](2d/..))  
  - `keypoint_stats.xlsx` summarizing per-file metrics
  - per-frame distance arrays and optional Excel summaries

Common CLI args:
- --threshold (center_of_mass threshold)
- --save_images, --save_annotations (save images and annotations as PNG files)
- --per_patient (group boxplots by patient)
- --no_sudden_movements / --threshold_sudden (pre/post processing for spike correction)

Example:
python 2d/test_models.py --threshold 0.875 --save_images --save_annotations --per_patient

See: [`test_models.process_h5_file_single`](2d/test_models.py) and the `main()` in the same file for batch loops and final Excel assembly.

---

## Important directories and purpose

- augmentations/: augmentation utilities used at training time ([2d/augmentations/aug0.py](2d/augmentations/aug0.py)).
- callbacks/: training callbacks (early stopping, lr scheduling) used by the training loop.
- cleanvision/: helper to visually inspect and filter the dataset using an external library (PIL / Imagelab).
- dataloader/: dataset creation, division and preprocessing helpers ([2d/dataloader/dataset_division.py](2d/dataloader/dataset_division.py), [2d/dataloader/preprocessing.py](2d/dataloader/preprocessing.py)).
- losses/: losses used to train keypoint regressors ([2d/losses/distances.py](2d/losses/distances.py)).
- models/: network architectures and initialization helpers.
- postprocessing/: Kalman filtering, COM extraction, heartbeat detection, index calculators.
- results/: aggregated Excel, plots, boxplots and separate results folders for different evaluation choices.
- model_weights/: storage for best model checkpoints from experiments.
- runs/: experiment folders generated during training.

---

## Typical workflows & tips

1. Train or fine-tune a model
   - Use [`2d/training.py`](2d/training.py).
   - Verify dataset format (npz for 2D training, `KeypointDataset` expects `images` and `keypoints`).
   - Monitor validation metrics and wandb output (wandb args available).

2. Run batch inference & indices
   - Prepare an Excel listing patient IDs / paths (script expects `path` column).
   - Use [`2d/indices_prediction.py`](2d/indices_prediction.py) to compute indices and save a `best_unet.xlsx` (or similar).
   - If you want per-file visual checks, add `--save_images` and set `--images_path`.

3. Validate results
   - Use [`2d/test_models.py`](2d/test_models.py) for in-depth per-file diagnostics (distance histograms, boxplots).
   - Run [`2d/statystical_analysis.py`](2d/statystical_analysis.py) to compute Bland–Altman stats and correlations vs the manual labels.

Common pitfalls:
- Check center-of-mass threshold (`--threshold` / `args.thresh`) — mismatch between training and inference threshold may cause missing or shifted points.
- Ensure model architecture args at inference match those used during training (`--depth`, `--filters`, `--residuals`).
- HDF5 structure must follow what the scripts expect (and same format):
  - tissue frames: `tissue/data`, tissue times: `tissue/times`
  - ECG: `ecg/ecg_data`, `ecg/ecg_times`
  - pixel size: `tissue/pixelsize`
- If using GPU, ensure CUDA environment is set; the code selects CUDA automatically if available.

---

## Where to look first in code
- Model & forward pass: [2d/models/models.py](2d/models/models.py)
- Prediction → keypoints extraction: [2d/indices_prediction.py](2d/indices_prediction.py) (`predict_indices`)
- Index formulas and utilities: [2d/postprocessing/indices_calculation.py](2d/postprocessing/indices_calculation.py)
- Bland–Altman and plotting: [2d/statystical_analysis.py](2d/statystical_analysis.py)