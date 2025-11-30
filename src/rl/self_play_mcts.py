"""
Minimal self-play + MCTS training loop (AlphaZero-style, simplified).
This is a baseline scaffold: single process, replay buffer, no distributed actors.
"""

import argparse
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    planes = np.zeros((13, 8, 8), dtype=np.float32)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            idx = (p.color * 6) + (p.piece_type - 1)
            planes[idx, sq // 8, sq % 8] = 1.0
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    return torch.from_numpy(planes)


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(13, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 4672),  # legal move upper bound
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        p = self.policy_head(h)
        v = self.value_head(h)
        return p, v


@dataclass
class Node:
    prior: float
    visit: int = 0
    value_sum: float = 0.0
    children: Dict[str, "Node"] = None
    move: chess.Move = None

    def value(self) -> float:
        return self.value_sum / self.visit if self.visit > 0 else 0.0


class MCTS:
    def __init__(self, net: PolicyValueNet, device: torch.device, sims: int = 64, c_puct: float = 1.4):
        self.net = net
        self.device = device
        self.sims = sims
        self.c_puct = c_puct

    def run(self, board: chess.Board) -> List[float]:
        root = Node(prior=1.0, children={})
        self._expand(root, board)
        for _ in range(self.sims):
            b_copy = board.copy()
            node = root
            path = []
            # select
            while node.children:
                key, node = self._select(node)
                b_copy.push(chess.Move.from_uci(key))
                path.append(node)
            # expand/evaluate
            if not b_copy.is_game_over():
                self._expand(node, b_copy)
                value = self._evaluate(b_copy)
            else:
                res = b_copy.result()
                value = 0.0 if res == "1/2-1/2" else (1.0 if res == "1-0" else -1.0)
            # backup
            for n in path:
                n.visit += 1
                n.value_sum += value
                value = -value
        # build policy target
        visits = np.zeros(4672, dtype=np.float32)
        legal = list(board.legal_moves)
        for mv in legal:
            key = mv.uci()
            if key in root.children:
                visits[legal.index(mv)] = root.children[key].visit
        if visits.sum() > 0:
            visits /= visits.sum()
        return visits.tolist()

    def _select(self, node: Node) -> Tuple[str, Node]:
        total = math.sqrt(sum(child.visit for child in node.children.values()) + 1)
        best, best_score = None, -1e9
        for key, child in node.children.items():
            u = self.c_puct * child.prior * total / (1 + child.visit)
            q = child.value()
            score = q + u
            if score > best_score:
                best_score = score
                best = (key, child)
        return best

    def _expand(self, node: Node, board: chess.Board):
        logits, _ = self._forward(board)
        legal = list(board.legal_moves)
        priors = torch.softmax(logits[: len(legal)], dim=0).detach().cpu().numpy()
        node.children = {}
        for mv, p in zip(legal, priors):
            node.children[mv.uci()] = Node(prior=float(p), children={}, move=mv)

    def _evaluate(self, board: chess.Board) -> float:
        _, v = self._forward(board)
        return float(v.item())

    def _forward(self, board: chess.Board) -> Tuple[torch.Tensor, torch.Tensor]:
        t = board_to_tensor(board).unsqueeze(0).to(self.device)
        logits, value = self.net(t)
        return logits.squeeze(0), value.squeeze(0)


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


def self_play(net: PolicyValueNet, mcts: MCTS, device: torch.device, games: int) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    samples = []
    for _ in range(games):
        b = chess.Board()
        states = []
        policies = []
        players = []
        while not b.is_game_over():
            pi = torch.tensor(mcts.run(b), dtype=torch.float32)
            states.append(board_to_tensor(b))
            policies.append(pi)
            players.append(1 if b.turn == chess.WHITE else -1)
            legal = list(b.legal_moves)
            idx = torch.multinomial(pi[: len(legal)], 1).item()
            b.push(legal[idx])
        res = b.result()
        z = 0.0 if res == "1/2-1/2" else (1.0 if res == "1-0" else -1.0)
        for s, p, pl in zip(states, policies, players):
            samples.append((s, p, z * pl))
    return samples


def train_step(net: PolicyValueNet, optimizer, batch, device: torch.device, c2: float = 1.0):
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


def main():
    parser = argparse.ArgumentParser(description="Self-play RL with MCTS.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--games-per-epoch", type=int, default=10)
    parser.add_argument("--mcts-sims", type=int, default=64)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-cap", type=int, default=50000)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PolicyValueNet(channels=args.channels).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    mcts = MCTS(net, device=device, sims=args.mcts_sims)
    buffer = ReplayBuffer(capacity=args.buffer_cap)

    for epoch in range(1, args.epochs + 1):
        samples = self_play(net, mcts, device=device, games=args.games_per_epoch)
        for s, p, v in samples:
            buffer.add((s, p, v))
        if len(buffer) >= args.batch_size:
            batch = buffer.sample(args.batch_size)
            loss, lp, lv = train_step(net, optimizer, batch, device=device)
            print(f"Epoch {epoch}: loss={loss:.4f} (p={lp:.4f}, v={lv:.4f}), buffer={len(buffer)}")
        if epoch % args.save_every == 0:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state": net.state_dict(), "arch": "pv_conv", "channels": args.channels},
                args.save_dir / f"rl_pv_epoch_{epoch}.pt",
            )


if __name__ == "__main__":
    main()
