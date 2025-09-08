import os
import json
import argparse
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data_loader import DataLoader
from src.volume_dataset import VolumeDataset, create_dataloader
from src.model import build_unet_3d, dice_coefficient_torch, iou_coefficient_torch, dice_loss_torch


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_split(split: dict, out_dir: str):
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump(split, f, indent=2)


def save_config(cfg: dict, out_dir: str):
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "train_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def evaluate(model, loader, device):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss = dice_loss_torch(y, pred)
            dice = dice_coefficient_torch(y, pred)
            iou = iou_coefficient_torch(y, pred)
            bs = x.size(0)
            val_loss += loss.item() * bs
            val_dice += dice.item() * bs
            val_iou += iou.item() * bs
            count += bs
    if count == 0:
        return {"loss": 0.0, "dice": 0.0, "iou": 0.0}
    return {"loss": val_loss / count, "dice": val_dice / count, "iou": val_iou / count}



def main(args):
    set_global_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.out_dir, f"run_{timestamp}")
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dl = DataLoader(
        data_dir=args.data_dir,
        wmin=args.wmin,
        wmax=args.wmax,
        add_channel=True
    )
    split = dl.split(ratios=(args.train_ratio, args.val_ratio, args.test_ratio), seed=args.seed)
    save_split(split, out_dir)

    train_ds = VolumeDataset(
        split["train"], dl,
        augment=True,
        p_flip=0.5, p_rot90=0.5, p_jitter=args.p_jitter, jitter_sigma=args.jitter_sigma,
        seed=args.seed
    )
    val_ds = VolumeDataset(split["val"], dl, augment=False)
    test_ds = VolumeDataset(split["test"], dl, augment=False)

    num_workers = min(4, os.cpu_count() or 0)
    train_loader = create_dataloader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = create_dataloader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    test_loader = create_dataloader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    model = build_unet_3d(
        input_shape=(64, 64, 64, 1),
        base_filters=args.base_filters,
        reg_l2=args.reg_l2,
        p_dropout_enc=args.dropout_enc,
        p_dropout_dec=args.dropout_dec,
        p_dropout_bot=args.dropout_bot,
        use_transpose=args.use_transpose
    ).to(device)

    lr = args.lr if args.lr is not None else 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=args.reg_l2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True)

    csv_path = os.path.join(out_dir, "training_log.csv")
    tb_dir = os.path.join("results", "logs", "tb", os.path.basename(out_dir))
    ensure_dir(tb_dir)
    print(f"Tensorboard logging -> {tb_dir}")
    writer = SummaryWriter(log_dir=tb_dir)

    save_config(vars(args), out_dir)

    best_val_loss = float('inf')
    patience_counter = 0
    ckpt_path = os.path.join(out_dir, "best_model.pt")

    with open(csv_path, 'w') as fcsv:
        fcsv.write("epoch,train_loss,val_loss,val_dice,val_iou\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        n_samples = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = dice_loss_torch(y, pred)
            loss.backward()
            optimizer.step()

            dice = dice_coefficient_torch(y, pred)
            iou = iou_coefficient_torch(y, pred)
            bs = x.size(0)
            running_loss += loss.item() * bs
            running_dice += dice.item() * bs
            running_iou += iou.item() * bs
            n_samples += bs

        train_loss = running_loss / max(1, n_samples)
        train_dice = running_dice / max(1, n_samples)
        train_iou = running_iou / max(1, n_samples)

        val_metrics = evaluate(model, val_loader, device)
        val_loss, val_dice, val_iou = val_metrics["loss"], val_metrics["dice"], val_metrics["iou"]

        scheduler.step(val_loss)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('IoU/train', train_iou, epoch)
        writer.add_scalar('IoU/val', val_iou, epoch)

        with open(csv_path, 'a') as fcsv:
            fcsv.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_dice:.6f},{val_iou:.6f}\n")

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'config': vars(args),
            }, ckpt_path)
            print(f"Saved new best model to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print("Early stopping triggered.")
                break

    writer.close()

    final_path = os.path.join(out_dir, "final_model.pt")
    torch.save({'model_state_dict': model.state_dict(), 'config': vars(args)}, final_path)

    # -------- Evaluate on test --------
    print("\nEvaluating on TEST split...")
    test_metrics = evaluate(model, test_loader, device)
    print("TEST metrics:", test_metrics)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)


#CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D Lung Segmentation U-Net")
    parser.add_argument("--data_dir", type=str, default="data", help="Dataset root (patient subfolders)")
    parser.add_argument("--out_dir", type=str, default="results/models", help="Output directory")


    parser.add_argument("--wmin", type=int, default=-1000, help="HU window min")
    parser.add_argument("--wmax", type=int, default=300, help="HU window max")

    
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio",   type=float, default=0.15)
    parser.add_argument("--test_ratio",  type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)


    parser.add_argument("--base_filters", type=int, default=16)
    parser.add_argument("--reg_l2", type=float, default=1e-5)
    parser.add_argument("--dropout_enc", type=float, default=0.10)
    parser.add_argument("--dropout_dec", type=float, default=0.10)
    parser.add_argument("--dropout_bot", type=float, default=0.10)
    parser.add_argument("--use_transpose", action="store_true", help="Use ConvTranspose3d instead of Upsample+Conv1x1")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")

    parser.add_argument("--p_jitter", type=float, default=0.2, help="Probability of intensity jitter")
    parser.add_argument("--jitter_sigma", type=float, default=0.03, help="Std of gaussian noise for jitter")

    args = parser.parse_args()
    main(args)
