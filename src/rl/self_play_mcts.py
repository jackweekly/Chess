"""
Parallel self-play loop using ParallelMCTS.
Runs N games simultaneously to maximize GPU batch efficiency.
"""

import argparse
import json
import random
from collections import deque
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
from tqdm import tqdm

import chess
import numpy as np
import torch
import torch.optim as optim

from src.models.alphazero import AlphaZeroNet
from src.rl.parallel_mcts import ParallelMCTS
from src.rl.encoders import get_input_tensor
from src.rl.action_encoding import ActionEncoder

# Action size for 8x8x73
ACTION_SIZE = 4672


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buf = deque(maxlen=capacity)

    def add(self, sample):
        self.buf.append(sample)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        states, policies, values = zip(*batch)
        return torch.stack(states), torch.stack(policies), torch.tensor(values, dtype=torch.float32)

    def __len__(self):
        return len(self.buf)


def self_play_parallel(
    net: AlphaZeroNet,
    mcts: ParallelMCTS,
    num_games: int,
    temperature_moves: int = 30,
    epoch: int = 0,
) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    encoder = mcts.encoder

    boards = [chess.Board() for _ in range(num_games)]
    histories = [[b.copy(stack=False)] for b in boards]

    game_data = [{"states": [], "policies": [], "players": []} for _ in range(num_games)]
    finished_games = [False] * num_games
    results = [0.0] * num_games

    pbar = tqdm(desc="Parallel Self-Play", unit="step")

    step = 0
    while not all(finished_games):
        pis = mcts.search(boards)
        for i in range(num_games):
            if finished_games[i]:
                continue
            b = boards[i]
            pi = pis[i]

            state, _ = get_input_tensor(b, histories[i])
            game_data[i]["states"].append(state)
            game_data[i]["policies"].append(torch.tensor(pi, dtype=torch.float32))
            game_data[i]["players"].append(1 if b.turn == chess.WHITE else -1)

            legal_indices = []
            legal_moves = list(b.legal_moves)
            for m in legal_moves:
                try:
                    legal_indices.append(encoder.encode(m, b))
                except Exception:
                    pass

            if not legal_indices:
                finished_games[i] = True
                continue

            legal_pi = pi[legal_indices]
            if legal_pi.sum() <= 1e-6:
                legal_pi = np.ones_like(legal_pi) / len(legal_pi)
            else:
                legal_pi /= legal_pi.sum()

            if len(game_data[i]["states"]) < temperature_moves:
                choice = np.random.choice(len(legal_indices), p=legal_pi)
            else:
                choice = np.argmax(legal_pi)

            move_idx = legal_indices[choice]
            move = encoder.decode(move_idx, b)
            b.push(move)
            histories[i].append(b.copy(stack=False))

            if b.is_game_over():
                finished_games[i] = True
                res = b.result()
                if res == "1-0":
                    results[i] = 1.0
                elif res == "0-1":
                    results[i] = -1.0
                else:
                    results[i] = 0.0
        step += 1
        pbar.update(1)
        if step % 10 == 0:
            finished_cnt = sum(finished_games)
            current_wr = 0.0
            if finished_cnt > 0:
                wins = sum(1 for r in results if r == 1.0)
                current_wr = wins / finished_cnt
            log_metrics(epoch=epoch, win_rate=current_wr, loss=0.0)
        if step > 400:
            break

    pbar.close()

    all_samples = []
    for i in range(num_games):
        z = results[i]
        for s, p, pl in zip(game_data[i]["states"], game_data[i]["policies"], game_data[i]["players"]):
            all_samples.append((s, p, z * pl))
    return all_samples


def train_step(net, optimizer, batch, device):
    states, pi_targets, v_targets = batch
    states = states.to(device)
    pi_targets = pi_targets.to(device)
    v_targets = v_targets.to(device)

    optimizer.zero_grad()
    pi_logits, v_pred = net(states)

    log_pi = torch.log_softmax(pi_logits, dim=1)
    loss_pi = -(pi_targets * log_pi).sum(dim=1).mean()
    loss_v = torch.nn.MSELoss()(v_pred.squeeze(), v_targets)

    loss = loss_pi + loss_v
    loss.backward()
    optimizer.step()

    return loss.item(), loss_pi.item(), loss_v.item()


def log_metrics(epoch: int, win_rate: float, loss: float):
    entry = {
        "ts": datetime.now().isoformat(),
        "epoch": epoch,
        "win": int(win_rate * 100),
        "games": 1,
        "loss_val": loss,
    }
    path = Path("data/eval/history.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(path.read_text()) if path.exists() else []
    except Exception:
        data = []
    data.append(entry)
    path.write_text(json.dumps(data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--games-per-epoch", type=int, default=16)
    parser.add_argument("--mcts-sims", type=int, default=800)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-cap", type=int, default=100000)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--load-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AlphaZeroNet(input_planes=119, channels=args.channels, blocks=args.blocks).to(device)

    if args.load_checkpoint and args.load_checkpoint.exists():
        print(f"Loading checkpoint: {args.load_checkpoint}")
        ckpt = torch.load(args.load_checkpoint, map_location=device)
        net.load_state_dict(ckpt.get("model_state", ckpt), strict=False)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    mcts = ParallelMCTS(net, num_games=args.games_per_epoch, sims=args.mcts_sims, device=device)
    buffer = ReplayBuffer(capacity=args.buffer_cap)

    print(f"Starting Training on {device}...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}: Self-Playing {args.games_per_epoch} games...")
        samples = self_play_parallel(net, mcts, num_games=args.games_per_epoch, epoch=epoch)

        for s in samples:
            buffer.add(s)

        print(f"Buffer size: {len(buffer)}. Training...")

        total_loss = 0
        steps = 0
        train_steps = max(1, len(samples) // max(1, (args.batch_size // 4)))

        if len(buffer) > args.batch_size:
            for _ in range(train_steps):
                batch = buffer.sample(args.batch_size)
                l, lp, lv = train_step(net, optimizer, batch, device)
                total_loss += l
                steps += 1

            avg_loss = total_loss / steps if steps else 0.0
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}")
            log_metrics(epoch, 0.5, avg_loss)

        if epoch % args.save_every == 0:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": net.state_dict(),
                    "epoch": epoch,
                    "arch": "alphazero",
                },
                args.save_dir / "rl_latest.pt",
            )


if __name__ == "__main__":
    main()
