import math
from typing import List, Tuple

import chess
import numpy as np
import torch

from src.rl.action_encoding import ActionEncoder
from src.rl.encoders import get_input_tensor


class MCTSNode:
    def __init__(self, prior: float = 1.0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # move_idx -> MCTSNode
        self.expanded = False

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


class ParallelMCTS:
    """
    Batched leaf-evaluation MCTS across multiple games for better GPU utilization.
    """

    def __init__(self, model, num_games: int, sims: int = 800, c_puct: float = 1.0, device: str = "cpu"):
        self.model = model
        self.num_games = num_games
        self.sims = sims
        self.c_puct = c_puct
        self.device = device
        self.encoder = ActionEncoder()
        self.roots = [None] * num_games

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        best_score = -1e9
        best_action = -1
        best_child = None
        sqrt_n = math.sqrt(max(node.visit_count, 1e-6))
        for action_idx, child in node.children.items():
            q = child.value()
            u = self.c_puct * child.prior * sqrt_n / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action_idx
                best_child = child
        return best_action, best_child

    def search(self, boards: List[chess.Board]) -> List[np.ndarray]:
        # initialize roots
        for i, board in enumerate(boards):
            if self.roots[i] is None:
                self.roots[i] = MCTSNode(prior=1.0)

        for _ in range(self.sims):
            leaves = []
            paths = []
            for i in range(self.num_games):
                node = self.roots[i]
                b = boards[i].copy()
                path = []
                while node.expanded and node.children:
                    action_idx, child = self._select_child(node)
                    path.append((node, action_idx))
                    mv = self.encoder.decode(action_idx, b)
                    if mv is None or mv not in b.legal_moves:
                        break
                    b.push(mv)
                    node = child
                leaves.append((i, node, b))
                paths.append(path)

            # batch inference
            tensors = []
            for _, _, b in leaves:
                t, _ = get_input_tensor(b, [b])
                tensors.append(t)
            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                logits, values = self.model(batch)
            probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
            values_batch = values.cpu().numpy()

            # expand and backup
            for leaf_idx, (game_i, node, b) in enumerate(leaves):
                if b.is_game_over():
                    res = b.result()
                    if res == "1-0":
                        value = 1.0 if b.turn == chess.WHITE else -1.0
                    elif res == "0-1":
                        value = -1.0 if b.turn == chess.WHITE else 1.0
                    else:
                        value = 0.0
                else:
                    node.expanded = True
                    legal = list(b.legal_moves)
                    priors = {}
                    total_p = 0.0
                    for mv in legal:
                        try:
                            idx = self.encoder.encode(mv, b)
                        except Exception:
                            continue
                        p = float(probs_batch[leaf_idx, idx])
                        priors[idx] = p
                        total_p += p
                    if total_p > 0:
                        for idx in priors:
                            priors[idx] /= total_p
                    for idx, p in priors.items():
                        node.children[idx] = MCTSNode(prior=p)
                    value = float(values_batch[leaf_idx].item())

                # backup
                node.visit_count += 1
                node.value_sum += value
                path = paths[leaf_idx]
                for parent, _ in reversed(path):
                    value = -value
                    parent.visit_count += 1
                    parent.value_sum += value

        # return visit distributions
        out = []
        for root in self.roots:
            visits = np.zeros(8 * 8 * 73, dtype=np.float32)
            if root and root.children:
                for idx, child in root.children.items():
                    visits[idx] = child.visit_count
                if visits.sum() > 0:
                    visits /= visits.sum()
            out.append(visits)
        return out
