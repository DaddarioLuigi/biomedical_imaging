# 3D Lung Segmentation Challenge – Student Submission

This repository contains my solution for the **3D Lung Segmentation Challenge**.
It includes a clean training pipeline (TensorFlow/Keras), data loading & preprocessing for 3D CT volumes, a compact **3D U-Net** with anti-overfitting additions, an online-augmentation **Keras `Sequence`**, and a minimal **inference** script.

---

## Project Structure

```
lung_segmentation_project/
│
├── data/                      # not included in repo
│   ├── patient_001/
│   │   ├── ct.nii.gz
│   │   └── lung.nii.gz
│   └── ...
│
├── src/
│   ├── data_loader.py         # DataLoader class: HU window + norm + split
│   ├── sequence.py            # Keras Sequence with random aug per epoch
│   ├── model.py               # 3D U-Net (+ L2, Dropout, stable upsampling)
│   ├── train.py               # training script w/ callbacks & logging
│   ├── inference.py           # inference: ct.nii.gz -> mask.nii.gz
│   └── viz.py (optional)      # simple overlays for figures
│
├── results/
│   ├── models/                # best_model.keras, final_model.keras, logs
│   └── preds/                 # saved predictions (optional)
│
├── report/
│   └── report.pdf             # 2–3 page report (methods+results)
│
├── requirements.txt
└── README.md
```

---

## Setup

**Windows/Linux (generic pip)**

```bash
python -m venv .venv
# Windows: py -3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Apple Silicon macOS (M1/M2/M3) — local dev**

```bash
# with Miniforge/Conda
conda create -n keras python=3.11 -y
conda activate keras
conda install -y tensorflow-macos jupyter numpy matplotlib scikit-image
conda install -y -c conda-forge nibabel
```

**requirements.txt**

```txt
numpy
matplotlib
scikit-image
nibabel
tensorflow
```

> On macOS use `tensorflow-macos` locally; for the submission we keep generic `tensorflow` to ensure portability on Windows/Linux.

---

## Dataset 

Each patient folder contains:

* `ct.nii.gz` (3D CT volume, already resized to **64×64×64** voxels)
* `lung.nii.gz` (binary mask of lungs, same shape)

Place all patient folders under `data/`.



##  Preprocessing (equal to the provided notebooks)

* **HU window**: clip CT to `[-1000, 300]` (configurable).
* **Normalization**: min–max scaling to **\[0, 1]**.
* **Mask binarization**: `> 0 → 1`.
* **Shapes**: `(64, 64, 64, 1)` tensors.

Implemented in `src/data_loader.py` (`DataLoader._apply_hu_window`, `_normalize_minmax`).

---

## Model

`src/model.py` implements a **3D U-Net** equivalent to the baseline with small improvements:

**Baseline (as in the notebook)**

* Encoder/decoder with residual shortcuts (downsampled and upsampled skip-adds).
* Final `Conv3D(1)` + `sigmoid` for per-voxel probabilities.

**NEW – overfitting & stability**

* `L2` regularization on all `Conv3D` (`--reg_l2`).
* `Dropout` in encoder/decoder and bottleneck (`--dropout_*`).
* Switchable upsampling: default **`UpSampling3D + Conv1×1`** (artifact-free) vs `Conv3DTranspose` (`--use_transpose`).
* Training with **Dice loss**, metrics **Dice** and **IoU**.

---

## Online Data Augmentation

`src/sequence.py` defines a **Keras `Sequence`** that loads cases on-the-fly and applies random, synchronized augmentations to CT & mask at each epoch:

* Random flips along D/H/W axes
* Random rotations by 90°, 180°, 270° around (D,H), (H,W), or (D,W)
* Optional intensity jitter (Gaussian noise) on CT in \[0,1]

This addresses overfitting and satisfies the assignment’s “random data augmentation each epoch via a Keras Sequence”.

---

##  Training

**Default command**

```bash
python -m src.train --data_dir data --out_dir results/models
```

**Common options**

```bash
python -m src.train --data_dir data --out_dir results/models \
  --base_filters 16 --reg_l2 1e-5 \
  --dropout_enc 0.10 --dropout_dec 0.10 --dropout_bot 0.10 \
  --epochs 150 --batch_size 2 \
  --p_jitter 0.2 --jitter_sigma 0.03
```

Artifacts saved for reproducibility:

* `split.json`, `train_config.json`, `training_log.csv`
* `best_model.keras` (by `val_loss`), `final_model.keras`
* `test_metrics.json` (Dice/IoU on held-out test)

Callbacks: `ReduceLROnPlateau`, `EarlyStopping`, `ModelCheckpoint`, `CSVLogger`.

---

## Inference

Predict a mask for a patient folder (containing `ct.nii.gz`) **or** a single CT file:

```bash
python -m src.inference \
  --model results/models/run_YYYYMMDD-HHMMSS/best_model.keras \
  --input data/patient_001 \
  --output_mask results/preds/patient_001_mask.nii.gz \
  --output_prob results/preds/patient_001_prob.nii.gz \
  --thr 0.5
```

The script applies the same HU window + normalization and saves NIfTI using the input CT affine/header.


## Reproducibility

* Fixed random seed (`--seed`).
* Persisted artifacts: `split.json`, `train_config.json`, logs, metrics, and model weights.
* Deterministic preprocessing & lightweight augmentations.

---

## Optional Enhancements

* **Hybrid loss**: `0.5 * DiceLoss + 0.5 * BinaryCrossentropy`
* **Test-Time Augmentation (TTA)**: average predictions over a few flips
* **Elastic deformation** (mild, 3D-aware) in the `Sequence`

