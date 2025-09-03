# src/train.py
# Training script minimale con callback e riproducibilità

import os
import json
import argparse
import random
import numpy as np
import tensorflow as tf

from datetime import datetime
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
)

from src.data_loader import DataLoader
from src.sequence import VolumeSequence
from src.model import build_unet_3d, dice_coefficient, iou_coefficient


# ---------------------------
# Utilità
# ---------------------------
def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def get_callbacks(out_dir: str):
    ensure_dir(out_dir)
    ckpt_path = os.path.join(out_dir, "best_model.keras")
    csv_path  = os.path.join(out_dir, "training_log.csv")

    cbs = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1),
        CSVLogger(csv_path)
    ]
    return cbs

def save_split(split: dict, out_dir: str):
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump(split, f, indent=2)

def save_config(cfg: dict, out_dir: str):
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "train_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


# ---------------------------
# Main training
# ---------------------------
def main(args):
    set_global_seed(args.seed)

    # cartelle output
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.out_dir, f"run_{timestamp}")
    ensure_dir(out_dir)

    # -------- Data --------
    dl = DataLoader(
        data_dir=args.data_dir,
        wmin=args.wmin,
        wmax=args.wmax,
        add_channel=True
    )
    split = dl.split(ratios=(args.train_ratio, args.val_ratio, args.test_ratio), seed=args.seed)
    save_split(split, out_dir)

    # Sequence (augment solo sul train)
    train_seq = VolumeSequence(
        split["train"], dl,
        batch_size=args.batch_size, shuffle=True, augment=True,
        p_flip=0.5, p_rot90=0.5, p_jitter=args.p_jitter, jitter_sigma=args.jitter_sigma,
        seed=args.seed
    )
    val_seq = VolumeSequence(
        split["val"], dl,
        batch_size=args.batch_size, shuffle=False, augment=False
    )
    test_seq = VolumeSequence(
        split["test"], dl,
        batch_size=args.batch_size, shuffle=False, augment=False
    )

    # -------- Model --------
    model = build_unet_3d(
        input_shape=(64, 64, 64, 1),
        base_filters=args.base_filters,
        reg_l2=args.reg_l2,                   # NEW: L2 (già nel model)
        p_dropout_enc=args.dropout_enc,       # NEW: Dropout encoder
        p_dropout_dec=args.dropout_dec,       # NEW: Dropout decoder
        p_dropout_bot=args.dropout_bot,       # NEW: Dropout bottleneck
        use_transpose=args.use_transpose      # NEW: switch upsampling
    )

    # Override LR se richiesto da CLI
    if args.lr is not None:
        model.optimizer.learning_rate = args.lr

    # salviamo anche la config
    save_config(vars(args), out_dir)

    # -------- Train --------
    callbacks = get_callbacks(out_dir)
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Salva il modello finale (oltre al best)
    model.save(os.path.join(out_dir, "final_model.keras"))

    # -------- Evaluate su test --------
    print("\nEvaluating on TEST split...")
    test_metrics = model.evaluate(test_seq, return_dict=True, verbose=1)
    print("TEST metrics:", test_metrics)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Nota: per l’inference userai src/inference.py separato.


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D Lung Segmentation U-Net")

    # paths
    parser.add_argument("--data_dir", type=str, default="data", help="Cartella dataset (sottocartelle paziente)")
    parser.add_argument("--out_dir", type=str, default="results/models", help="Cartella di output")

    # preprocessing
    parser.add_argument("--wmin", type=int, default=-1000, help="HU window min")
    parser.add_argument("--wmax", type=int, default=300, help="HU window max")

    # split
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio",   type=float, default=0.15)
    parser.add_argument("--test_ratio",  type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    # model
    parser.add_argument("--base_filters", type=int, default=16)
    parser.add_argument("--reg_l2", type=float, default=1e-5)
    parser.add_argument("--dropout_enc", type=float, default=0.10)
    parser.add_argument("--dropout_dec", type=float, default=0.10)
    parser.add_argument("--dropout_bot", type=float, default=0.10)
    parser.add_argument("--use_transpose", action="store_true", help="Usa Conv3DTranspose invece di UpSampling3D+Conv")

    # train
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate dell'optimizer")

    # augmentation
    parser.add_argument("--p_jitter", type=float, default=0.2, help="Probabilità di intensity jitter sul CT")
    parser.add_argument("--jitter_sigma", type=float, default=0.03, help="Std del rumore gaussiano per jitter")

    args = parser.parse_args()
    main(args)
