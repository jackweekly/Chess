import argparse
import time
from pathlib import Path

import optuna
import torch
import pysnooper
from optuna.trial import Trial
from supervised_baseline import (
    ConvPolicy,
    MoveMLP,
    ParquetMoveIterableDataset,
    load_label_encoder,
    train,
)
from torch.utils.data import DataLoader


def build_model(trial: Trial, n_classes: int, device: torch.device):
    arch = trial.suggest_categorical("arch", ["conv", "mlp"])
    if arch == "conv":
        channels = trial.suggest_int("channels", 64, 160, step=32)
        blocks = trial.suggest_int("blocks", 2, 6)
        model = ConvPolicy(channels=channels, blocks=blocks, n_classes=n_classes)
    else:
        hidden = trial.suggest_int("hidden", 256, 1024, step=256)
        model = MoveMLP(hidden=hidden, n_classes=n_classes)
    return model.to(device), arch


def objective(trial: Trial, args) -> float:
    data_dir = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_classes = load_label_encoder(data_dir)
    n_classes = len(label_classes)

    batch_size = trial.suggest_categorical("batch_size", [256, 512, 768, 1024])
    lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
    read_batch = trial.suggest_categorical("read_batch", [4096, 8192, 16384, 32768])
    num_workers = trial.suggest_int("num_workers", 2, 8)

    model, arch = build_model(trial, n_classes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_ds = ParquetMoveIterableDataset(
        data_dir, split="train", rank=0, world_size=1, read_batch_size=read_batch, max_files=args.max_train_files
    )
    val_ds = ParquetMoveIterableDataset(
        data_dir, split="val", rank=0, world_size=1, read_batch_size=read_batch, max_files=args.max_val_files
    )
    # Use a small subset for speed
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=args.num_workers, pin_memory=True)
    loaders = {"train": train_loader, "val": val_loader}

    start = time.time()
    # Single epoch
    train(
        model,
        loaders,
        device=device,
        n_classes=n_classes,
        epochs=1,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        lr=lr,
        debug=args.debug,
    )
    elapsed = time.time() - start

    # Evaluate quickly on a few val batches
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (xb, yb) in enumerate(val_loader):
            xb, yb = xb.to(device), yb.to(device)
            mask = (yb >= 0) & (yb < n_classes)
            if mask.sum() == 0:
                continue
            xb = xb[mask]
            yb = yb[mask]
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(xb)
            if i >= args.val_batches - 1:
                break
    acc = correct / max(total, 1)
    trial.report(acc, step=0)
    # Combine acc and speed (optional)
    if args.throughput_weight > 0:
        score = acc + args.throughput_weight * (1.0 / max(elapsed, 1e-3))
    else:
        score = acc
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna tuner for supervised policy.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/supervised"))
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--val-batches", type=int, default=10, help="Val batches to sample for quick eval.")
    parser.add_argument("--max-train-batches", type=int, default=200, help="Cap train batches per trial.")
    parser.add_argument("--max-val-batches", type=int, default=50, help="Cap val batches per trial.")
    parser.add_argument("--max-train-files", type=int, default=1, help="Limit train shards to speed up trials.")
    parser.add_argument("--max-val-files", type=int, default=1, help="Limit val shards to speed up trials.")
    parser.add_argument("--throughput-weight", type=float, default=0.0, help="Weight to add throughput into score.")
    parser.add_argument("--snoop-file", type=Path, default=None, help="Optional pysnooper log file for tuning.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for tuning.")
    parser.add_argument("--debug", action="store_true", help="Print debug info from training loops.")
    args = parser.parse_args()

    def run():
        print(f"[tune] Starting study with {args.trials} trials, max_train_batches={args.max_train_batches}, max_val_batches={args.max_val_batches}, workers={args.num_workers}")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, args), n_trials=args.trials, show_progress_bar=False)
        print("Best trial:", study.best_trial.value)
        print("Params:", study.best_trial.params)

    if args.snoop_file:
        args.snoop_file.parent.mkdir(parents=True, exist_ok=True)
        with pysnooper.snoop(args.snoop_file):
            run()
    else:
        run()


if __name__ == "__main__":
    main()
