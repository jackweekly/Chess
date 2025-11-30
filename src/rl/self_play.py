"""
Lightweight self-play loop to bootstrap reinforcement learning.

This uses a simple policy network over legal moves to illustrate the training flow.
It is intentionally small; extend with stronger architectures and MCTS for better play.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

import chess
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    planes = np.zeros((13, 64), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            offset = (piece.color * 6) + (piece.piece_type - 1)
            planes[offset, square] = 1.0
    planes[12, :] = 1.0 if board.turn == chess.WHITE else 0.0
    return torch.from_numpy(planes.ravel())


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(13 * 64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4672),  # upper bound on legal moves in chess
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@dataclass
class Trajectory:
    states: List[torch.Tensor]
    actions: List[int]
    rewards: List[float]
    legal_maps: List[List[chess.Move]]


def sample_action(logits: torch.Tensor, legal_moves: List[chess.Move]) -> Tuple[int, chess.Move]:
    probs = torch.softmax(logits[: len(legal_moves)], dim=0)
    idx = torch.multinomial(probs, 1).item()
    return idx, legal_moves[idx]


def play_game(policy: PolicyNet, device: torch.device, reward_win: float = 1.0, reward_draw: float = 0.0) -> Trajectory:
    board = chess.Board()
    states: List[torch.Tensor] = []
    actions: List[int] = []
    rewards: List[float] = []
    legal_maps: List[List[chess.Move]] = []

    while not board.is_game_over():
        state = board_to_tensor(board).to(device)
        logits = policy(state)
        legal_moves = list(board.legal_moves)
        legal_maps.append(legal_moves)
        move_idx, move = sample_action(logits, legal_moves)

        states.append(state)
        actions.append(move_idx)

        board.push(move)

    result = board.result()
    if result == "1-0":
        rewards = [reward_win if i % 2 == 0 else -reward_win for i in range(len(actions))]
    elif result == "0-1":
        rewards = [-reward_win if i % 2 == 0 else reward_win for i in range(len(actions))]
    else:
        rewards = [reward_draw for _ in actions]

    return Trajectory(states=states, actions=actions, rewards=rewards, legal_maps=legal_maps)


def reinforce_step(policy: PolicyNet, optimizer: optim.Optimizer, device: torch.device, episodes: int = 4) -> float:
    policy.train()
    logprobs = []
    returns = []
    for _ in range(episodes):
        traj = play_game(policy, device=device)
        G = 0.0
        for r in reversed(traj.rewards):
            G = r + 0.99 * G
            returns.insert(0, G)

        for state, action, legal in zip(traj.states, traj.actions, traj.legal_maps):
            logits = policy(state)
            log_prob = torch.log_softmax(logits[: len(legal)], dim=0)[action]
            logprobs.append(log_prob)

    returns_tensor = torch.tensor(returns, device=device)
    returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-6)
    loss = -(torch.stack(logprobs) * returns_tensor).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main() -> None:
    # Distributed init (torchrun). Single-GPU fallback otherwise.
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNet().to(device)
    if dist.is_initialized():
        policy = DDP(policy, device_ids=[device], output_device=device)

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    epochs = int(os.environ.get("EPOCHS", 5))
    episodes_per_epoch = int(os.environ.get("EPISODES_PER_EPOCH", 4))
    for epoch in range(epochs):
        loss = reinforce_step(policy, optimizer, device=device, episodes=episodes_per_epoch)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Epoch {epoch} loss: {loss:.4f}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
