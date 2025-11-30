import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import pyarrow.parquet as pq
from tqdm import tqdm
import math
import pysnooper


class ParquetMoveIterableDataset(IterableDataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        rank: int = 0,
        world_size: int = 1,
        read_batch_size: int = 8192,
        max_files: Optional[int] = None,
    ):
        self.files = sorted(Path(data_dir).glob(f"{split}_*.parquet"))
        if max_files is not None:
            self.files = self.files[:max_files]
        if not self.files:
            raise FileNotFoundError(f"No parquet shards found for split='{split}' in {data_dir}")
        self.files = self.files[rank::max(world_size, 1)]
        self.read_batch_size = read_batch_size
        self.total_rows = 0
        for f in self.files:
            try:
                pf = pq.ParquetFile(f)
                self.total_rows += pf.metadata.num_rows
            except Exception:
                continue
        self.total_batches = math.ceil(self.total_rows / self.read_batch_size) if self.total_rows else None

    def __iter__(self):
        worker = get_worker_info()
        if worker is not None:
            files = self.files[worker.id :: worker.num_workers]
        else:
            files = self.files

        for file_path in files:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.read_batch_size):
                feats = batch.column("features").to_pylist()
                labels = batch.column("label").to_pylist()
                if not feats:
                    continue
                for f, l in zip(feats, labels):
                    yield torch.tensor(f, dtype=torch.float32), torch.tensor(l, dtype=torch.long)


class MoveMLP(nn.Module):
    def __init__(self, hidden: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13 * 64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.act(out)
        return out


class ConvPolicy(nn.Module):
    def __init__(self, channels: int, blocks: int, n_classes: int, input_channels: int = 13):
        super().__init__()
        self.input_channels = input_channels
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.resblocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, C*64)
        x = x.view(-1, self.input_channels, 8, 8)
        out = self.stem(x)
        out = self.resblocks(out)
        logits = self.head(out)
        return logits


class AlphaZeroNet(ConvPolicy):
    """ResNet policy head reused for RL with an added value head."""

    def __init__(self, channels: int = 128, blocks: int = 10, n_classes: int = 4096, input_channels: int = 103):
        super().__init__(channels=channels, blocks=blocks, n_classes=n_classes, input_channels=input_channels)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, self.input_channels, 8, 8)
        out = self.stem(x)
        out = self.resblocks(out)
        pi = self.head(out)
        v = self.value_head(out)
        return pi, v


def load_label_encoder(output_dir: Path) -> np.ndarray:
    return np.load(output_dir / "label_encoder_classes.npy", allow_pickle=False)


def train(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    n_classes: int,
    epochs: int = 5,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    lr: float = 1e-3,
    non_blocking: bool = False,
    debug: bool = False,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else torch.amp.GradScaler("cpu")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        train_iter = tqdm(
            loaders["train"],
            disable=dist.is_initialized() and dist.get_rank() != 0,
            desc=f"Epoch {epoch+1} [train]",
            total=max_train_batches,
        )
        for batch_idx, (xb, yb) in enumerate(train_iter):
            xb, yb = xb.to(device, non_blocking=non_blocking), yb.to(device, non_blocking=non_blocking)
            valid_mask = (yb >= 0) & (yb < n_classes)
            if valid_mask.sum() == 0:
                continue
            xb = xb[valid_mask]
            yb = yb[valid_mask]
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(xb)
            total_samples += len(xb)
            if debug and (batch_idx + 1) % 50 == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
                train_iter.write(f"debug batch {batch_idx+1}: loss {loss.item():.4f}, samples {total_samples}")
            if max_train_batches is not None and (batch_idx + 1) >= max_train_batches:
                break

        model.eval()
        correct = torch.tensor(0, device=device)
        total = torch.tensor(0, device=device)
        top5 = torch.tensor(0, device=device)
        with torch.no_grad():
            for batch_idx, (xb, yb) in enumerate(loaders["val"]):
                xb, yb = xb.to(device, non_blocking=non_blocking), yb.to(device, non_blocking=non_blocking)
                valid_mask = (yb >= 0) & (yb < n_classes)
                if valid_mask.sum() == 0:
                    continue
                xb = xb[valid_mask]
                yb = yb[valid_mask]
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum()
                # top-5
                topk = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
                top5 += (topk == yb.unsqueeze(1)).any(dim=1).sum()
                total += torch.tensor(len(xb), device=device)
                if debug and (batch_idx + 1) % 20 == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"debug val batch {batch_idx+1}: correct {correct.item()} total {total.item()}")
                if max_val_batches is not None and (batch_idx + 1) >= max_val_batches:
                    break

        if dist.is_initialized():
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            dist.all_reduce(top5, op=dist.ReduceOp.SUM)

        avg_loss = total_loss / max(total_samples, 1)
        acc = (correct.item() / max(total.item(), 1)) if total.item() else 0.0
        acc5 = (top5.item() / max(total.item(), 1)) if total.item() else 0.0
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f} val_acc={acc:.3f} val_top5={acc5:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised move predictor baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/supervised"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (IterableDataset-friendly)")
    parser.add_argument("--arch", type=str, choices=["mlp", "conv"], default="conv", help="Model architecture")
    parser.add_argument("--channels", type=int, default=128, help="Conv channels for conv arch")
    parser.add_argument("--blocks", type=int, default=6, help="Residual blocks for conv arch")
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"), help="Checkpoint output dir")
    parser.add_argument("--read-batch-size", type=int, default=8192, help="Rows per parquet read batch")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Cap training batches for speed")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Cap validation batches for speed")
    parser.add_argument("--max-train-files", type=int, default=None, help="Limit number of train shards to read")
    parser.add_argument("--max-val-files", type=int, default=None, help="Limit number of val shards to read")
    parser.add_argument("--debug", action="store_true", help="Print periodic debug info during training")
    parser.add_argument("--snoop-file", type=Path, default=None, help="Optional pysnooper log file")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--cudnn-benchmark", action="store_true", help="Enable cudnn benchmark for speed")
    parser.add_argument("--non-blocking", action="store_true", help="Use non_blocking transfers to GPU")
    args = parser.parse_args()

    label_classes = load_label_encoder(args.data_dir)
    n_classes = len(label_classes)

    # Distributed init if launched via torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.arch == "mlp":
        model = MoveMLP(hidden=args.hidden, n_classes=n_classes).to(device)
    else:
        model = ConvPolicy(channels=args.channels, blocks=args.blocks, n_classes=n_classes).to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[device], output_device=device)

    if args.cudnn_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    train_dataset = ParquetMoveIterableDataset(
        args.data_dir,
        split="train",
        rank=rank,
        world_size=world_size,
        read_batch_size=args.read_batch_size,
        max_files=args.max_train_files,
    )
    val_dataset = ParquetMoveIterableDataset(
        args.data_dir,
        split="val",
        rank=rank,
        world_size=world_size,
        read_batch_size=args.read_batch_size,
        max_files=args.max_val_files,
    )

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
    }

    train_fn = lambda: train(
        model,
        loaders,
        device=device,
        n_classes=n_classes,
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        lr=args.lr,
        non_blocking=args.non_blocking,
        debug=args.debug,
    )

    if args.snoop_file:
        with pysnooper.snoop(args.snoop_file):
            train_fn()
    else:
        train_fn()

    if (not dist.is_initialized()) or dist.get_rank() == 0:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "model_state": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
            "arch": args.arch,
            "channels": args.channels,
            "blocks": args.blocks,
            "classes": label_classes,
            "epochs": args.epochs,
        }
        torch.save(ckpt, args.save_dir / "supervised_policy.pt")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
