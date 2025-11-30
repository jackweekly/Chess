import gymnasium as gym
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.rl.chess_env import ChessEnv

def verify_env():
    print("Initializing ChessEnv...")
    env = ChessEnv(render_mode="ansi")
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    assert obs.shape == (13, 8, 8)
    assert env.action_space.n == 4096
    
    print("Running random agent loop...")
    terminated = False
    truncated = False
    steps = 0
    
    while not terminated and not truncated and steps < 100:
        # Sample random action (likely illegal)
        # To make it run longer, let's try to pick a legal move if possible
        # But for verification of the 'illegal move' logic, random is fine.
        # We'll try to parse legal moves from info to pick a valid one for a few steps
        
        legal_ucis = info["legal_moves"]
        if legal_ucis:
            import chess
            move_uci = np.random.choice(legal_ucis)
            move = chess.Move.from_uci(move_uci)
            action = move.from_square * 64 + move.to_square
        else:
            action = env.action_space.sample()
            
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        
        if steps % 10 == 0:
            print(f"Step {steps}, Reward: {reward}")
            # print(env.render())
            
    print(f"Finished after {steps} steps. Final Reward: {reward}")
    print("Verification Successful!")

if __name__ == "__main__":
    verify_env()
