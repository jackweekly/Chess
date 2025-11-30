import chess
import numpy as np
import torch
from typing import List


def get_input_tensor(board: chess.Board, history: List[chess.Board] | None = None) -> torch.Tensor:
    """
    Build a 119-plane AlphaZero-style tensor.
    - 8 historical boards, each with 12 planes (6 pieces x 2 colors) = 96
    - Metadata planes:
        4 castling rights (WK, WQ, BK, BQ)
        1 side to move
        1 move count (normalized)
        2 repetition flags (>=1, >=2)
        15 zero padding planes to reach 119
    """
    if history is None:
        history = [board]

    # Keep last 8 states; pad with oldest if needed
    history_states = list(history)[-8:]
    if len(history_states) < 8:
        history_states = [history_states[0]] * (8 - len(history_states)) + history_states

    planes = []

    # Piece planes, most recent last in history for consistency
    for b in history_states:
        board_planes = np.zeros((12, 8, 8), dtype=np.float32)
        for color in (chess.WHITE, chess.BLACK):
            for piece_type in range(1, 7):
                for sq in b.pieces(piece_type, color):
                    idx = (0 if color == chess.WHITE else 6) + (piece_type - 1)
                    board_planes[idx, sq // 8, sq % 8] = 1.0
        planes.append(board_planes)

    meta = np.zeros((8, 8, 8), dtype=np.float32)
    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        meta[0] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        meta[1] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        meta[2] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        meta[3] = 1
    # Side to move
    meta[4] = 1 if board.turn == chess.WHITE else 0
    # Move count normalized
    meta[5] = min(len(board.move_stack) / 200.0, 1.0)
    # Repetition flags (approx)
    meta[6] = 1 if board.is_repetition(2) else 0
    meta[7] = 1 if board.is_repetition(3) else 0

    planes.append(meta)

    stack = np.concatenate(planes, axis=0)
    if stack.shape[0] < 119:
        pad = np.zeros((119 - stack.shape[0], 8, 8), dtype=np.float32)
        stack = np.concatenate([stack, pad], axis=0)
    elif stack.shape[0] > 119:
        stack = stack[:119]

    return torch.from_numpy(stack).float()
