"""
Simple self-play reinforcement learning loop.
This is a minimal example: single process self-play with a policy/value head,
on-policy updates (REINFORCE with value baseline), no MCTS.
Use as a starting point and iterate.
"""

import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    planes = np.zeros((13, 64), dtype=np.float32)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            offset = (p.color * 6) + (p.piece_type - 1)
            planes[offset, sq] = 1.0
    planes[12, :] = 1.0 if board.turn == chess.WHITE else 0.0
    return torch.from_numpy(planes.ravel())


class PolicyValue(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(13 * 64, hidden),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, 4672)  # upper bound legal moves
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.policy(h), self.value(h)


@dataclass
class Trajectory:
    states: List[torch.Tensor]
    actions: List[int]
    legal_maps: List[List[chess.Move]]
    rewards: List[float]


def sample_action(logits: torch.Tensor, legal_moves: List[chess.Move]) -> Tuple[int, chess.Move, torch.Tensor]:
    logits = logits[: len(legal_moves)]
    probs = torch.softmax(logits, dim=0)
    idx = torch.multinomial(probs, 1).item()
    return idx, legal_moves[idx], torch.log(probs[idx] + 1e-8)


def play_game(model: PolicyValue, device: torch.device, reward_win: float = 1.0, reward_draw: float = 0.0) -> Trajectory:
    board = chess.Board()
    states: List[torch.Tensor] = []
    actions: List[int] = []
    legal_maps: List[List[chess.Move]] = []
    rewards: List[float] = []

    while not board.is_game_over():
        state = board_to_tensor(board).to(device)
        logits, _ = model(state)
        legal_moves = list(board.legal_moves)
        idx, mv, _ = sample_action(logits, legal_moves)
        states.append(state)
        actions.append(idx)
        legal_maps.append(legal_moves)
        board.push(mv)

    result = board.result()
    if result == "1-0":
        rewards = [reward_win if i % 2 == 0 else -reward_win for i in range(len(actions))]
    elif result == "0-1":
        rewards = [-reward_win if i % 2 == 0 else reward_win for i in range(len(actions))]
    else:
        rewards = [reward_draw for _ in actions]

    return Trajectory(states=states, actions=actions, legal_maps=legal_maps, rewards=rewards)


def reinforce_update(model: PolicyValue, trajs: List[Trajectory], optimizer: optim.Optimizer, device: torch.device, gamma: float = 0.99):
    logps = []
    returns = []
    values = []
    for traj in trajs:
        G = 0.0
        traj_returns = []
        for r in reversed(traj.rewards):
            G = r + gamma * G
            traj_returns.insert(0, G)
        for state, action, legal, Gt in zip(traj.states, traj.actions, traj.legal_maps, traj_returns):
            logits, val = model(state)
            lp = torch.log_softmax(logits[: len(legal)], dim=0)[action]
            logps.append(lp)
            values.append(val.squeeze())
            returns.append(torch.tensor(Gt, device=device))

    if not logps:
        return 0.0
    logps = torch.stack(logps)
    values = torch.stack(values)
    returns = torch.stack(returns)
    adv = returns - values.detach()
    policy_loss = -(logps * adv).mean()
    value_loss = nn.MSELoss()(values, returns)
    loss = policy_loss + 0.5 * value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description="Self-play RL (policy/value).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--games-per-epoch", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyValue(hidden=args.hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        trajs = [play_game(model, device=device) for _ in range(args.games_per_epoch)]
        loss = reinforce_update(model, trajs, optimizer, device=device, gamma=args.gamma)
        print(f"Epoch {epoch}: loss={loss:.4f}")
        if epoch % args.save_every == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({"model_state": model.state_dict()}, f"{args.save_dir}/rl_policy_{epoch}.pt")


if __name__ == "__main__":
    main()
