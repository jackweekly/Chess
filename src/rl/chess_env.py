import gymnasium as gym
import numpy as np
import chess
from gymnasium import spaces

class ChessEnv(gym.Env):
    """
    A Gymnasium environment for Chess.
    
    Action Space: Discrete(4096) representing (from_square * 64 + to_square).
    Observation Space: Box(0, 1, (13, 8, 8)) representing the board state.
    """
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.board = chess.Board()
        self.render_mode = render_mode
        
        # Action space: 64 * 64 = 4096 possible moves (from_square -> to_square)
        # We ignore underpromotions for simplicity (always promote to Queen)
        self.action_space = spaces.Discrete(64 * 64)
        
        # Observation space: 8x8 board with 13 planes
        # Planes 0-5: White pieces (P, N, B, R, Q, K)
        # Planes 6-11: Black pieces (P, N, B, R, Q, K)
        # Plane 12: 1 if White's turn, 0 if Black's turn
        self.observation_space = spaces.Box(low=0, high=1, shape=(13, 8, 8), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        return self._get_obs(), self._get_info()

    def step(self, action):
        # Decode action
        from_sq = action // 64
        to_sq = action % 64
        
        move = chess.Move(from_sq, to_sq)
        
        # Handle promotion (auto-queen)
        if (
            self.board.piece_type_at(from_sq) == chess.PAWN
            and chess.square_rank(to_sq) in [0, 7]
        ):
            move.promotion = chess.QUEEN

        # Check legality
        if move in self.board.legal_moves:
            self.board.push(move)
            terminated = self.board.is_game_over()
            truncated = False
            
            # Reward structure
            if terminated:
                result = self.board.result()
                if result == "1-0":
                    reward = 1.0 if self.board.turn == chess.BLACK else -1.0
                elif result == "0-1":
                    reward = 1.0 if self.board.turn == chess.WHITE else -1.0
                else:
                    reward = 0.0 # Draw
            else:
                reward = 0.0
                
            info = self._get_info()
            return self._get_obs(), reward, terminated, truncated, info
        else:
            # Illegal move
            # We can either return a large negative reward and terminate, 
            # or just return a negative reward and continue (ignoring the move).
            # For standard RL, ending the episode on illegal move is common to learn rules.
            return self._get_obs(), -10.0, True, False, {"error": "illegal_move"}

    def render(self):
        if self.render_mode == "ansi":
            return str(self.board)
        elif self.render_mode == "human":
            print(self.board)
        
    def close(self):
        pass

    def _get_obs(self):
        planes = np.zeros((13, 8, 8), dtype=np.float32)
        for sq in chess.SQUARES:
            p = self.board.piece_at(sq)
            if p:
                # 0-5 for White, 6-11 for Black
                layer = (p.piece_type - 1) + (0 if p.color == chess.WHITE else 6)
                row, col = divmod(sq, 8)
                planes[layer, row, col] = 1.0
        
        if self.board.turn == chess.WHITE:
            planes[12, :, :] = 1.0
            
        return planes

    def _get_info(self):
        return {
            "legal_moves": [m.uci() for m in self.board.legal_moves],
            "turn": "white" if self.board.turn == chess.WHITE else "black"
        }
