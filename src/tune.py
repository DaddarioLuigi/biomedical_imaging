import os
import json
from datetime import datetime

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ray import tune
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

from src.data_loader import DataLoader
from src.volume_dataset import VolumeDataset, create_dataloader
from src.model import build_unet_3d, dice_loss_torch, dice_coefficient_torch, iou_coefficient_torch


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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


def train_fn(config: dict):
    #Determinism across trials
    seed = int(config.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_dir = config.get("data_dir", "data")
    out_root = config.get("out_dir", "results/models")

    #Per-trial output dir
    trial_id = tune.get_trial_id() if tune.get_trial_id() is not None else datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(out_root, f"tune_{trial_id}")
    ensure_dir(out_dir)

    dl = DataLoader(
        data_dir=data_dir,
        wmin=int(config.get("wmin", -1000)),
        wmax=int(config.get("wmax", 300)),
        add_channel=True,
    )
    split = dl.split(
        ratios=(config.get("train_ratio", 0.7), config.get("val_ratio", 0.15), config.get("test_ratio", 0.15)),
        seed=seed,
    )

    train_ds = VolumeDataset(
        split["train"], dl,
        augment=True,
        p_flip=0.5, p_rot90=0.5,
        p_jitter=float(config.get("p_jitter", 0.2)), jitter_sigma=float(config.get("jitter_sigma", 0.03)),
        seed=seed,
    )
    val_ds = VolumeDataset(
        split["val"], dl,
        augment=False,
    )

    num_workers = min(4, os.cpu_count() or 0)
    train_loader = create_dataloader(train_ds, batch_size=int(config.get("batch_size", 2)), shuffle=True, num_workers=num_workers)
    val_loader = create_dataloader(val_ds, batch_size=int(config.get("batch_size", 2)), shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet_3d(
        input_shape=(64, 64, 64, 1),
        base_filters=int(config.get("base_filters", 16)),
        reg_l2=float(config.get("reg_l2", 1e-5)),
        p_dropout_enc=float(config.get("dropout_enc", 0.10)),
        p_dropout_dec=float(config.get("dropout_dec", 0.10)),
        p_dropout_bot=float(config.get("dropout_bot", 0.10)),
        use_transpose=bool(config.get("use_transpose", False)),
    ).to(device)

    #Optimizer & scheduler
    lr = float(config.get("lr", 1e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=float(config.get("reg_l2", 1e-5)))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=False)

    epochs = int(config.get("epochs", 60))
    best_val_loss = float('inf')
    csv_path = os.path.join(out_dir, "training_log.csv")
    with open(csv_path, 'w') as fcsv:
        fcsv.write("epoch,train_loss,val_loss,val_dice,val_iou\n")

    for epoch in range(1, epochs + 1):
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

        with open(csv_path, 'a') as fcsv:
            fcsv.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_dice:.6f},{val_iou:.6f}\n")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            ckpt_path = os.path.join(out_dir, "best_model.pt")
            torch.save({'model_state_dict': model.state_dict(), 'config': config, 'epoch': epoch, 'best_val_loss': best_val_loss}, ckpt_path)

        session.report({
            "val_loss": float(val_loss),
            "val_dice": float(val_dice),
            "val_iou": float(val_iou),
            "out_dir": out_dir,
        })


def run_tuning():
    #search space
    param_space = {
        "data_dir": tune.choice(["data"]),
        "out_dir": tune.choice(["results/models"]),
        "seed": 42,
        #model
        "base_filters": tune.choice([16, 24, 32]),
        "reg_l2": tune.loguniform(1e-6, 1e-3),
        "dropout_enc": tune.uniform(0.0, 0.3),
        "dropout_dec": tune.uniform(0.0, 0.3),
        "dropout_bot": tune.uniform(0.0, 0.3),
        "use_transpose": tune.choice([False, True]),
        #training
        "batch_size": tune.choice([1, 2, 3, 4]),
        "epochs": 60,
        "lr": tune.loguniform(3e-5, 3e-4),
        #augment
        "p_jitter": tune.uniform(0.0, 0.5),
        "jitter_sigma": tune.loguniform(0.01, 0.08),
        #split
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        #preprocess
        "wmin": -1000,
        "wmax": 300,
    }

    #Scheduler and searcher
    scheduler = ASHAScheduler(metric="val_loss", mode="min", max_t=60, grace_period=10, reduction_factor=2)
    search_alg = OptunaSearch(metric="val_loss", mode="min")

    tuner = tune.Tuner(
        tune.with_resources(train_fn, {"cpu": 4, "gpu": 0}),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=12,
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        run_config=tune.RunConfig(
            name="lung3d_hpo",
            local_dir="results/ray",
            stop=None,
        ),
    )

    results = tuner.fit()

    #Top results summary
    best_result = results.get_best_result(metric="val_loss", mode="min")
    summary = {
        "best_metrics": best_result.metrics,
        "best_out_dir": best_result.metrics.get("out_dir"),
        "best_config": best_result.config,
    }
    os.makedirs("results/ray", exist_ok=True)
    with open("results/ray/best_result.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    run_tuning() 