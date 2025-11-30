"""
Self-play + MCTS training loop (AlphaZero-style) using a ResNet policy/value network.
Supports 119-plane inputs, 4672-class move space (from supervised label encoder),
and batched leaf evaluation for faster MCTS.
"""

import argparse
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.alphazero import AlphaZeroNet
from src.rl.encoders import get_input_tensor


def load_move_encoder(path: Path) -> Tuple[List[str], Dict[str, int]]:
    classes = np.load(path, allow_pickle=True)
    idx_to_move = [str(x) for x in classes.tolist()]
    move_to_idx = {uci: i for i, uci in enumerate(idx_to_move)}
    return idx_to_move, move_to_idx


def mirror_move(move: chess.Move) -> chess.Move:
    """Mirror a move for black/white perspective flip."""
    return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), promotion=move.promotion)


@dataclass
class Node:
    prior: float
    visit: int = 0
    value_sum: float = 0.0
    children: Dict[str, "Node"] | None = None
    move: chess.Move | None = None

    def value(self) -> float:
        return self.value_sum / self.visit if self.visit > 0 else 0.0


class MCTS:
    def __init__(
        self,
        net: AlphaZeroNet,
        device: torch.device,
        sims: int = 64,
        c_puct: float = 1.4,
        move_to_idx: Dict[str, int] | None = None,
        idx_to_move: List[str] | None = None,
        batch_size: int = 64,
    ):
        self.net = net
        self.device = device
        self.sims = sims
        self.c_puct = c_puct
        self.move_to_idx = move_to_idx or {}
        self.idx_to_move = idx_to_move or []
        self.action_size = len(self.idx_to_move)
        self.batch_size = batch_size

    def _legal_indices(self, board: chess.Board) -> List[Tuple[chess.Move, int]]:
        pairs = []
        for mv in board.legal_moves:
            idx = self.move_to_idx.get(mv.uci())
            if idx is not None:
                pairs.append((mv, idx))
        return pairs

    def get_masked_logits(self, board: chess.Board, logits: torch.Tensor) -> torch.Tensor:
        """Mask illegal or unknown moves to large negative values."""
        mask = torch.full((self.action_size,), -1e9, device=logits.device, dtype=logits.dtype)
        pairs = self._legal_indices(board)
        if not pairs:
            return mask
        legal_indices = [idx for _, idx in pairs]
        mask[legal_indices] = logits[legal_indices]
        return mask

    def run(
        self,
        board: chess.Board,
        history: List[chess.Board] | None = None,
        sims: Optional[int] = None,
    ) -> List[float]:
        if self.action_size == 0:
            raise ValueError("MCTS missing move encoder (action_size=0)")

        root_history = [b.copy(stack=False) for b in (history or [board])]
        root = Node(prior=1.0, children={})

        # Initialize root expansion
        logits, flipped = self._forward(board, root_history)
        self._set_children_from_logits(root, board if not flipped else board.mirror(), logits, flipped)

        total_sims = sims if sims is not None else self.sims
        pending: List[Tuple[Node, chess.Board, List[chess.Board], List[Node]]] = []

        for sim in range(total_sims):
            b_copy = board.copy()
            history_states = list(root_history)
            node = root
            path_nodes = [root]

            # Selection
            while node.children:
                key, node = self._select(node)
                mv = chess.Move.from_uci(key)
                b_copy.push(mv)
                history_states.append(b_copy.copy(stack=False))
                path_nodes.append(node)

            # Terminal check
            if b_copy.is_game_over():
                res = b_copy.result()
                value = 0.0 if res == "1/2-1/2" else (1.0 if res == "1-0" else -1.0)
                self._backup(path_nodes, value)
            else:
                pending.append((node, b_copy, history_states, path_nodes))

            # Process batch
            if len(pending) >= self.batch_size or sim == total_sims - 1:
                self._process_leaves(pending)
                pending = []

        visits = np.zeros(self.action_size, dtype=np.float32)
        for uci, child in (root.children or {}).items():
            idx = self.move_to_idx.get(uci)
            if idx is not None:
                visits[idx] = child.visit
        total = visits.sum()
        if total > 0:
            visits /= total
        return visits.tolist()

    def get_best_move(
        self, board: chess.Board, history: List[chess.Board] | None = None, sims: Optional[int] = None
    ) -> str:
        visits = self.run(board, history=history, sims=sims)
        legal = list(board.legal_moves)
        best_move = None
        best_score = -1.0
        for mv in legal:
            idx = self.move_to_idx.get(mv.uci())
            if idx is None:
                continue
            score = visits[idx]
            if score > best_score:
                best_score = score
                best_move = mv
        if best_move is None:
            raise ValueError("No legal moves mapped in encoder")
        return best_move.uci()

    def _process_leaves(self, leaves: List[Tuple[Node, chess.Board, List[chess.Board], List[Node]]]):
        if not leaves:
            return
        tensors = []
        flips = []
        for _, b, h, _ in leaves:
            t, flipped = get_input_tensor(b, h)
            tensors.append(t)
            flips.append(flipped)
        batch = torch.stack([t.to(self.device) for t in tensors])
        logits_batch, values_batch = self.net(batch)
        for (node, b, _, path_nodes), logits, value, flipped in zip(leaves, logits_batch, values_batch, flips):
            self._set_children_from_logits(node, b if not flipped else b.mirror(), logits, flipped)
            self._backup(path_nodes, float(value.item()))

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

    def _set_children_from_logits(self, node: Node, board: chess.Board, logits: torch.Tensor, flipped: bool):
        masked = self.get_masked_logits(board if not flipped else board.mirror(), logits)
        pairs = self._legal_indices(board if not flipped else board.mirror())
        node.children = {}
        if not pairs:
            return
        priors = torch.softmax(masked, dim=0).detach().cpu().numpy()
        total_prior = 0.0
        for mv, idx in pairs:
            p = float(priors[idx])
            uci = mv.uci()
            if flipped:
                # mirror move back to original orientation
                uci = mirror_move(mv).uci()
            node.children[uci] = Node(prior=p, children={}, move=mv)
            total_prior += p
        if total_prior > 0:
            for child in node.children.values():
                child.prior /= total_prior

    def _backup(self, path_nodes: List[Node], value: float):
        for n in reversed(path_nodes):
            n.visit += 1
            n.value_sum += value
            value = -value

    def _forward(self, board: chess.Board, history: List[chess.Board]) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        t, flipped = get_input_tensor(board, history)
        t = t.unsqueeze(0).to(self.device)
        logits, value = self.net(t)
        return logits.squeeze(0), value.squeeze(0), flipped


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


def self_play(
    net: AlphaZeroNet,
    mcts: MCTS,
    device: torch.device,
    games: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    samples = []
    for _ in range(games):
        b = chess.Board()
        history_states: List[chess.Board] = [b.copy(stack=False)]
        states = []
        policies = []
        players = []
        move_count = 0
        while not b.is_game_over():
            pi = torch.tensor(mcts.run(b, history_states), dtype=torch.float32)
            state, _ = get_input_tensor(b, history_states)
            states.append(state)
            policies.append(pi)
            players.append(1 if b.turn == chess.WHITE else -1)
            legal = list(b.legal_moves)
            legal_indices = [mcts.move_to_idx.get(mv.uci()) for mv in legal]
            legal_pairs = [(mv, idx) for mv, idx in zip(legal, legal_indices) if idx is not None]
            if not legal_pairs:
                break
            probs = pi[[idx for _, idx in legal_pairs]]
            if probs.sum() <= 0:
                probs = torch.ones(len(legal_pairs), dtype=torch.float32) / len(legal_pairs)
            else:
                probs = probs / probs.sum()
            # Temperature scheduling: sample early, argmax later
            if move_count < 30:
                choice = torch.multinomial(probs, 1).item()
            else:
                choice = int(torch.argmax(probs).item())
            b.push(legal_pairs[choice][0])
            history_states.append(b.copy(stack=False))
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
    mask = (targets_p > 0).float()
    normalizer = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    loss_p = -((targets_p * logp * mask).sum(dim=1) / normalizer.squeeze()).mean()
    loss_v = nn.MSELoss()(values.squeeze(), targets_v)
    loss = loss_p + c2 * loss_v
    loss.backward()
    optimizer.step()
    return loss.item(), loss_p.item(), loss_v.item()


def main():
    parser = argparse.ArgumentParser(description="Self-play RL with batched MCTS.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--games-per-epoch", type=int, default=10)
    parser.add_argument("--mcts-sims", type=int, default=128)
    parser.add_argument("--mcts-batch", type=int, default=64)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-cap", type=int, default=50000)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--move-list", type=Path, default=Path("data/processed/supervised/label_encoder_classes.npy"))
    parser.add_argument("--init-ckpt", type=Path, default=None, help="Optional supervised checkpoint to warm start",)
    args = parser.parse_args()

    idx_to_move, move_to_idx = load_move_encoder(args.move_list)
    action_size = len(idx_to_move)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AlphaZeroNet(channels=args.channels, blocks=args.blocks, n_classes=action_size, input_channels=119)
    if args.init_ckpt and args.init_ckpt.exists():
        state = torch.load(args.init_ckpt, map_location="cpu")
        net.load_state_dict(state.get("model_state", state), strict=False)
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    mcts = MCTS(
        net,
        device=device,
        sims=args.mcts_sims,
        move_to_idx=move_to_idx,
        idx_to_move=idx_to_move,
        batch_size=args.mcts_batch,
    )
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
                {
                    "model_state": net.state_dict(),
                    "arch": "alphazero_resnet",
                    "channels": args.channels,
                    "blocks": args.blocks,
                    "moves": idx_to_move,
                },
                args.save_dir / f"rl_pv_epoch_{epoch}.pt",
            )


if __name__ == "__main__":
    main()
