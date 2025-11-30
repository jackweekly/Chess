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
from src.models.supervised_baseline import AlphaZeroNet


def move_to_index(move: chess.Move) -> int:
    """Map a move to a flat 4096 index (from_square * 64 + to_square). Promotions share the same index."""
    return move.from_square * 64 + move.to_square


def get_extensive_board_tensor(board: chess.Board, history: List[chess.Board] | None = None) -> torch.Tensor:
    """
    AlphaZero-style encoding with temporal context and metadata.
    - 8 historical board states (12 planes each: 6 pieces x 2 colors) = 96 planes
    - +7 metadata planes (castling rights, side to move, repetition flag, move count) = 103 planes total
    """
    if history is None:
        history = [board]

    # Keep the last 8 states; pad with the oldest if shorter.
    history_states = list(history)[-8:]
    if len(history_states) < 8:
        history_states = [history_states[0]] * (8 - len(history_states)) + history_states

    planes: list[np.ndarray] = []

    # Piece planes for each historical board.
    for b in history_states:
        board_planes = np.zeros((12, 8, 8), dtype=np.float32)
        for sq in chess.SQUARES:
            p = b.piece_at(sq)
            if p:
                idx = (p.color * 6) + (p.piece_type - 1)
                board_planes[idx, sq // 8, sq % 8] = 1.0
        planes.append(board_planes)

    # Metadata planes for the current board only.
    meta_planes = np.zeros((7, 8, 8), dtype=np.float32)
    if board.has_kingside_castling_rights(chess.WHITE):
        meta_planes[0, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        meta_planes[1, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        meta_planes[2, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        meta_planes[3, :, :] = 1
    if board.turn == chess.WHITE:
        meta_planes[4, :, :] = 1
    if board.is_repetition(2):
        meta_planes[5, :, :] = 1
    meta_planes[6, :, :] = min(len(board.move_stack) / 100.0, 1.0)

    planes.append(meta_planes)

    full_stack = np.concatenate(planes, axis=0)
    return torch.from_numpy(full_stack).float()

# Backwards-compatible alias; previous code referenced board_to_tensor.
board_to_tensor = get_extensive_board_tensor


# Backwards-compatible alias pointing to the stronger ResNet-based model.
PolicyValueNet = AlphaZeroNet


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

    def get_masked_logits(self, board: chess.Board, logits: torch.Tensor) -> torch.Tensor:
        """Mask illegal moves to a large negative value so softmax ignores them."""
        mask = torch.full_like(logits, -1e9)
        legal_indices = [move_to_index(m) for m in board.legal_moves]
        mask[legal_indices] = logits[legal_indices]
        return mask

    def run(self, board: chess.Board, history: List[chess.Board] | None = None, sims: int | None = None) -> List[float]:
        root_history = [b.copy(stack=False) for b in (history or [board])]
        root = Node(prior=1.0, children={})
        self._expand(root, board, root_history)
        total_sims = sims if sims is not None else self.sims
        for _ in range(total_sims):
            b_copy = board.copy()
            history_states = list(root_history)
            node = root
            path = []
            # select
            while node.children:
                key, node = self._select(node)
                b_copy.push(chess.Move.from_uci(key))
                history_states.append(b_copy.copy(stack=False))
                path.append(node)
            # expand/evaluate
            if not b_copy.is_game_over():
                self._expand(node, b_copy, history_states)
                value = self._evaluate(b_copy, history_states)
            else:
                res = b_copy.result()
                value = 0.0 if res == "1/2-1/2" else (1.0 if res == "1-0" else -1.0)
            # backup
            for n in path:
                n.visit += 1
                n.value_sum += value
                value = -value
        # build policy target
        visits = np.zeros(4096, dtype=np.float32)
        legal = list(board.legal_moves)
        for mv in legal:
            key = mv.uci()
            if key in root.children:
                visits[move_to_index(mv)] = root.children[key].visit
        total = visits.sum()
        if total > 0:
            visits /= total
        return visits.tolist()

    def get_best_move(self, board: chess.Board, history: List[chess.Board] | None = None, sims: int | None = None) -> str:
        """Runs MCTS and returns the UCI of the most visited legal move."""
        visits = self.run(board, history=history, sims=sims)
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal moves available")
        best_move = legal[0]
        best_score = -1.0
        for mv in legal:
            score = visits[move_to_index(mv)]
            if score > best_score:
                best_score = score
                best_move = mv
        return best_move.uci()

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

    def _expand(self, node: Node, board: chess.Board, history: List[chess.Board]):
        logits, _ = self._forward(board, history)
        masked_logits = self.get_masked_logits(board, logits)
        priors = torch.softmax(masked_logits, dim=0).detach().cpu().numpy()
        legal = list(board.legal_moves)
        node.children = {}
        for mv in legal:
            idx = move_to_index(mv)
            node.children[mv.uci()] = Node(prior=float(priors[idx]), children={}, move=mv)

    def _evaluate(self, board: chess.Board, history: List[chess.Board]) -> float:
        _, v = self._forward(board, history)
        return float(v.item())

    def _forward(self, board: chess.Board, history: List[chess.Board]) -> Tuple[torch.Tensor, torch.Tensor]:
        t = get_extensive_board_tensor(board, history).unsqueeze(0).to(self.device)
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
        history_states: List[chess.Board] = [b.copy(stack=False)]
        states = []
        policies = []
        players = []
        while not b.is_game_over():
            pi = torch.tensor(mcts.run(b, history_states), dtype=torch.float32)
            states.append(get_extensive_board_tensor(b, history_states))
            policies.append(pi)
            players.append(1 if b.turn == chess.WHITE else -1)
            legal = list(b.legal_moves)
            legal_indices = [move_to_index(mv) for mv in legal]
            legal_probs = pi[legal_indices]
            if legal_probs.sum() <= 0:
                legal_probs = torch.ones(len(legal), dtype=torch.float32) / len(legal)
            else:
                legal_probs = legal_probs / legal_probs.sum()
            idx = torch.multinomial(legal_probs, 1).item()
            b.push(legal[idx])
            history_states.append(b.copy(stack=False))
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
    # Only optimize over moves that have nonzero target probability (i.e., legal/visited).
    mask = (targets_p > 0).float()
    # Avoid division by zero if a row has all zeros (fallback to averaging over all moves).
    normalizer = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    loss_p = -((targets_p * logp * mask).sum(dim=1) / normalizer.squeeze()).mean()
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
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-cap", type=int, default=50000)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PolicyValueNet(channels=args.channels, blocks=args.blocks, n_classes=4096, input_channels=103).to(device)
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
