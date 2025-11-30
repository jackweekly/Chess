"""
Batched self-play loop with AlphaZero-style model, 73-plane policy head, and ActionEncoder mapping.
Leaf batching across multiple games for GPU efficiency.
"""

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.alphazero import AlphaZeroNet
from src.rl.action_encoding import ActionEncoder
from src.rl.encoders import get_input_tensor

ACTION_SIZE = 8 * 8 * 73  # 4672


@dataclass
class Node:
    prior: float
    visit: int = 0
    value_sum: float = 0.0
    children: dict | None = None

    def value(self) -> float:
        return self.value_sum / self.visit if self.visit > 0 else 0.0


class BatchedMCTS:
    def __init__(self, net: AlphaZeroNet, device: torch.device, sims: int = 128, c_puct: float = 1.0, batch_size: int = 32):
        self.net = net
        self.device = device
        self.sims = sims
        self.c_puct = c_puct
        self.encoder = ActionEncoder()
        self.batch_size = batch_size

    def _select(self, node: Node) -> int:
        total = sum(child.visit for child in node.children.values()) + 1
        best, best_score = None, -1e9
        for idx, child in node.children.items():
            u = self.c_puct * child.prior * (total ** 0.5) / (1 + child.visit)
            q = child.value()
            score = q + u
            if score > best_score:
                best_score = score
                best = idx
        return best

    def run(self, board: chess.Board, sims: int | None = None) -> np.ndarray:
        root = Node(prior=1.0, children={})
        logits, flipped = self._forward(board, [board])
        self._expand(root, board, logits, flipped)

        pending = []
        total_sims = sims or self.sims
        for sim in range(total_sims):
            b = board.copy()
            history = [b.copy(stack=False)]
            node = root
            path = [root]
            while node.children:
                action_idx = self._select(node)
                mv = self.encoder.decode(action_idx, b)
                if mv is None or mv not in b.legal_moves:
                    break
                b.push(mv)
                history.append(b.copy(stack=False))
                node = node.children[action_idx]
                path.append(node)
            if b.is_game_over():
                res = b.result()
                value = 0.0 if res == "1/2-1/2" else (1.0 if res == "1-0" else -1.0)
                self._backup(path, value)
            else:
                pending.append((node, b, history, path))
            if len(pending) >= self.batch_size or sim == total_sims - 1:
                self._process_leaves(pending)
                pending = []

        visits = np.zeros(ACTION_SIZE, dtype=np.float32)
        for idx, child in (root.children or {}).items():
            visits[idx] = child.visit
        total = visits.sum()
        if total > 0:
            visits /= total
        return visits

    def _process_leaves(self, leaves):
        if not leaves:
            return
        tensors = []
        flips = []
        boards = []
        for _, b, h, _ in leaves:
            t, flipped = get_input_tensor(b, h)
            tensors.append(t)
            flips.append(flipped)
            boards.append(b)
        batch = torch.stack([t.to(self.device) for t in tensors])
        with torch.no_grad():
            logits_batch, values_batch = self.net(batch)
        for (node, b, _, path), logits, value, flipped in zip(leaves, logits_batch, values_batch, flips):
            self._expand(node, b, logits, flipped)
            self._backup(path, float(value.item()))

    def _expand(self, node: Node, board: chess.Board, logits: torch.Tensor, flipped: bool):
        legal = list(board.legal_moves)
        priors = torch.softmax(logits, dim=0).detach().cpu().numpy()
        node.children = {}
        total_prior = 0.0
        for mv in legal:
            try:
                idx = self.encoder.encode(mv, board)
            except Exception:
                continue
            p = float(priors[idx])
            node.children[idx] = Node(prior=p, children={})
            total_prior += p
        if total_prior > 0:
            for child in node.children.values():
                child.prior /= total_prior
        node.visit += 1

    def _backup(self, path: List[Node], value: float):
        for n in reversed(path):
            n.visit += 1
            n.value_sum += value
            value = -value

    def _forward(self, board: chess.Board, history: List[chess.Board]) -> Tuple[torch.Tensor, bool]:
        t, flipped = get_input_tensor(board, history)
        logits, value = self.net(t.unsqueeze(0).to(self.device))
        return logits.squeeze(0), flipped


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


def self_play(net: AlphaZeroNet, mcts: BatchedMCTS, games: int, temperature_moves: int = 30) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    encoder = mcts.encoder
    samples = []
    for _ in range(games):
        b = chess.Board()
        history = [b.copy(stack=False)]
        states = []
        policies = []
        players = []
        move_count = 0
        while not b.is_game_over():
            pi = mcts.run(b)
            state, _ = get_input_tensor(b, history)
            states.append(state)
            policies.append(torch.tensor(pi, dtype=torch.float32))
            players.append(1 if b.turn == chess.WHITE else -1)
            # sample move
            legal = list(b.legal_moves)
            legal_idx = []
            for mv in legal:
                try:
                    idx = encoder.encode(mv, b)
                    legal_idx.append(idx)
                except Exception:
                    continue
            if not legal_idx:
                break
            probs = torch.tensor(pi[legal_idx], dtype=torch.float32)
            if probs.sum() <= 0:
                probs = torch.ones(len(legal_idx)) / len(legal_idx)
            else:
                probs = probs / probs.sum()
            if move_count < temperature_moves:
                choice = torch.multinomial(probs, 1).item()
            else:
                choice = int(torch.argmax(probs).item())
            mv = encoder.decode(legal_idx[choice], b)
            if mv is None:
                break
            b.push(mv)
            history.append(b.copy(stack=False))
            move_count += 1
        res = b.result()
        z = 0.0 if res == "1/2-1/2" else (1.0 if res == "1-0" else -1.0)
        for s, p, pl in zip(states, policies, players):
            samples.append((s, p, z * pl))
    return samples


def train_step(net: AlphaZeroNet, optimizer, batch, device: torch.device, c2: float = 1.0):
    states, targets_p, targets_v = batch
    states = states.to(device)
    targets_p = targets_p.to(device)
    targets_v = targets_v.to(device)
    optimizer.zero_grad()
    logits, values = net(states)
    logp = torch.log_softmax(logits, dim=1)
    loss_p = -(targets_p * logp).sum(dim=1).mean()
    loss_v = nn.MSELoss()(values.squeeze(), targets_v)
    loss = loss_p + c2 * loss_v
    loss.backward()
    optimizer.step()
    return loss.item(), loss_p.item(), loss_v.item()


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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--games-per-epoch", type=int, default=8)
    parser.add_argument("--mcts-sims", type=int, default=128)
    parser.add_argument("--mcts-batch", type=int, default=32)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-cap", type=int, default=50000)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AlphaZeroNet(input_planes=119, channels=args.channels, blocks=args.blocks).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    mcts = BatchedMCTS(net, device=device, sims=args.mcts_sims, batch_size=args.mcts_batch)
    buffer = ReplayBuffer(capacity=args.buffer_cap)

    for epoch in range(1, args.epochs + 1):
        samples = self_play(net, mcts, games=args.games_per_epoch)
        for s, p, v in samples:
            buffer.add((s, p, v))
        if len(buffer) >= args.batch_size:
            batch = buffer.sample(args.batch_size)
            loss, lp, lv = train_step(net, optimizer, batch, device=device)
            log_metrics(epoch, win_rate=0.0, loss=loss)
            print(f"Epoch {epoch}: loss={loss:.4f} (p={lp:.4f}, v={lv:.4f}), buffer={len(buffer)}")
        if epoch % args.save_every == 0:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": net.state_dict(),
                    "arch": "alphazero_73",
                    "channels": args.channels,
                    "blocks": args.blocks,
                },
                args.save_dir / f"rl_pv_epoch_{epoch}.pt",
            )


if __name__ == "__main__":
    main()
