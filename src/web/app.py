import asyncio
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import chess
import chess.engine
import numpy as np
import random
import torch
import torch.nn as nn
import torch.serialization
from collections import OrderedDict
import functools
import time
from datetime import datetime
from copy import copy
import contextlib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import threading
from src.rl.self_play_mcts import PolicyValueNet, MCTS, ReplayBuffer, board_to_tensor

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Chess Assistant")

# Global game state (single session for now)
board = chess.Board()
engine = None
engine_name = "none"
engine_mode = "stockfish"  # "stockfish" or "model"
search_depth = 12
search_time = None  # seconds (float) if set
policy_model: Optional[nn.Module] = None
policy_device: torch.device = torch.device("cpu")
move_to_id: Dict[str, int] = {}
id_to_move: List[str] = []
eval_history_path = Path("data/eval/history.json")
match_board: Optional[chess.Board] = None
match_running: bool = False
match_task: Optional[asyncio.Task] = None
model_as_white: bool = True
policy_optimizer: Optional[torch.optim.Optimizer] = None
# RL trainer state
rl_thread: Optional[threading.Thread] = None
rl_stop = threading.Event()
rl_status: str = "idle"
rl_ckpt_path = Path("checkpoints/rl_latest.pt")
rl_buffer = ReplayBuffer(capacity=50000)


def maybe_load_engine() -> Optional[chess.engine.SimpleEngine]:
    """
    Try to load Stockfish if available; otherwise return None.
    Looks for a compiled binary under third_party/stockfish/src/stockfish or on PATH.
    """
    candidates = [
        Path("third_party/stockfish/src/stockfish"),
        Path("third_party/stockfish/stockfish"),
        Path("/usr/games/stockfish"),
    ]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            try:
                return chess.engine.SimpleEngine.popen_uci(str(cand))
            except Exception:
                continue
    # Fall back to PATH
    try:
        return chess.engine.SimpleEngine.popen_uci("stockfish")
    except Exception:
        return None


def reset_game(fen: Optional[str] = None) -> None:
    global board, game_trajectory
    board = chess.Board(fen) if fen else chess.Board()
    game_trajectory = []


def board_to_tensor(b: chess.Board) -> torch.Tensor:
    planes = np.zeros((13, 64), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = b.piece_at(sq)
        if piece:
            offset = (piece.color * 6) + (piece.piece_type - 1)
            planes[offset, sq] = 1.0
    planes[12, :] = 1.0 if b.turn == chess.WHITE else 0.0
    return torch.from_numpy(planes.ravel())


def san_stack_from_moves(moves: List[chess.Move]) -> List[str]:
    temp_board = chess.Board()
    san_moves: List[str] = []
    for mv in moves:
        san_moves.append(temp_board.san(mv))
        temp_board.push(mv)
    return san_moves


async def suggest_move() -> dict:
    """Return best move suggestion and score if engine is available."""
    if engine_mode == "model":
        mv = policy_move()
        return {"move": mv, "score": None, "source": "model", "pv": [], "info": {}}
    if engine is None:
        return {"move": None, "score": None, "source": engine_mode, "pv": [], "info": {}}
    try:
        limit = chess.engine.Limit(depth=search_depth, time=search_time)
        info = await engine.play(board, limit=limit, info=chess.engine.INFO_ALL)
        score = info.info.get("score")
        pv_moves = [m.uci() for m in info.info.get("pv", [])] if info.info else []
        cp = score.white().score(mate_score=10_000) if score else None
        details = {
            "depth": info.info.get("depth"),
            "nodes": info.info.get("nodes"),
            "nps": info.info.get("nps"),
            "time": info.info.get("time"),
        }
        return {
            "move": info.move.uci() if info.move else None,
            "score": cp,
            "source": engine_name,
            "pv": pv_moves,
            "info": details,
        }
    except Exception:
        return {"move": None, "score": None, "source": "error", "pv": [], "info": {}}


@app.on_event("startup")
async def startup_event() -> None:
    global engine, engine_name
    engine = maybe_load_engine()
    if engine:
        try:
            engine_name = engine.id.get("name", "engine")
        except Exception:
            engine_name = "engine"
    
    global rl_thread
    rl_stop.clear()
    rl_thread = threading.Thread(target=background_train_loop, daemon=True)
    rl_thread.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global engine
    if engine is not None:
        with contextlib.suppress(Exception):
            engine.quit()
        engine = None
    
    rl_stop.set()
    if rl_thread:
        rl_thread.join(timeout=1.0)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/state")
async def get_state():
    suggestion = await suggest_move()
    san_moves = san_stack_from_moves(list(board.move_stack))
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "move_stack": [m.uci() for m in board.move_stack],
        "san_stack": san_moves,
        "legal_moves": [m.uci() for m in board.legal_moves],
        "suggestion": suggestion,
        "is_game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "engine": {
            "name": engine_name,
            "available": engine is not None,
            "mode": engine_mode,
            "depth": search_depth,
            "time": search_time,
            "temperature": model_temperature,
            "epsilon": model_epsilon,
        },
        "model_as_white": model_as_white,
    }


@app.post("/api/set_side")
async def api_set_side(payload: dict):
    global model_as_white
    side = payload.get("model_as_white")
    if side is None:
        raise HTTPException(status_code=400, detail="Missing model_as_white")
    model_as_white = bool(side)
    return await get_state()

@app.post("/api/reset")
async def api_reset(payload: dict):
    fen = payload.get("fen")
    try:
        reset_game(fen)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {e}")
    return await get_state()


# Online Learning State
learning_enabled: bool = True
game_trajectory: List[Tuple[torch.Tensor, int]] = []
current_model_path: Path = Path("checkpoints/rl_latest.pt")
rl_buffer = ReplayBuffer(capacity=100000)

def save_policy() -> None:
    """Save the current model state to disk."""
    if policy_model is None:
        return
    try:
        current_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": policy_model.state_dict(),
            "classes": id_to_move,
            "arch": "mlp",
            "hidden": 768
        }, current_model_path)
    except Exception as e:
        print(f"[Auto-Save] Error: {e}")

def background_train_loop():
    """Continuously train the model from the replay buffer."""
    global policy_model, policy_optimizer
    batch_size = 32
    print("[Background Train] Started")
    while not rl_stop.is_set():
        if len(rl_buffer) >= batch_size and policy_model is not None and policy_optimizer is not None:
            try:
                states, actions, rewards = rl_buffer.sample(batch_size)
                states = states.to(policy_device)
                actions = actions.to(policy_device)
                rewards = rewards.to(policy_device)

                policy_model.train()
                policy_optimizer.zero_grad()
                logits = policy_model(states)
                
                # Policy Gradient Loss: -mean(log_prob * reward)
                logp = torch.log_softmax(logits, dim=1)
                sel = logp[torch.arange(batch_size, device=policy_device), actions]
                loss = -(sel * rewards).mean()
                
                loss.backward()
                policy_optimizer.step()
                policy_model.eval()
                
                # Auto-save occasionally? Or every step? User liked live updates.
                # Let's save every 10 steps to avoid disk thrashing but keep it "live"
                if random.random() < 0.1:
                    save_policy()
            except Exception as e:
                print(f"[Background Train] Error: {e}")
        time.sleep(0.1)
    print("[Background Train] Stopped")

def train_step_supervised(b: chess.Board, move: chess.Move) -> None:
    """Imitation Learning: Push sample to buffer with reward 1.0."""
    if not learning_enabled:
        return
    
    try:
        u = move.uci()
        if u not in move_to_id:
            return
        target_idx = move_to_id[u]
        
        tensor = board_to_tensor(b) # CPU tensor
        # Reward 1.0 for imitation
        rl_buffer.add((tensor, torch.tensor(target_idx, dtype=torch.long), 1.0))
        # print(f"[Imitation] Added sample to buffer. Size: {len(rl_buffer)}")
    except Exception as e:
        print(f"[Imitation] Error: {e}")

def train_step_rl(result: str) -> None:
    """Reinforcement Learning: Push trajectory to buffer with game outcome reward."""
    global game_trajectory
    if not learning_enabled or not game_trajectory:
        game_trajectory = []
        return

    try:
        # Calculate reward based on result
        if result == "1-0":
            reward = 1.0 if model_as_white else -1.0
        elif result == "0-1":
            reward = -1.0 if model_as_white else 1.0
        else:
            reward = 0.0

        if reward == 0.0:
            game_trajectory = []
            return

        for state, action_idx in game_trajectory:
            # state is already CPU tensor from board_to_tensor
            rl_buffer.add((state, torch.tensor(action_idx, dtype=torch.long), reward))
        
        print(f"[RL] Game Over ({result}). Added {len(game_trajectory)} samples to buffer. Reward: {reward}")
    except Exception as e:
        print(f"[RL] Error: {e}")
    finally:
        game_trajectory = []

@app.post("/api/move")
async def api_move(payload: dict):
    move_str = payload.get("move")
    if not move_str:
        raise HTTPException(status_code=400, detail="Missing move")
    try:
        move = board.parse_uci(move_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid move format")
    if move not in board.legal_moves:
        raise HTTPException(status_code=400, detail="Illegal move")
    
    # Determine if this is a User move or Model move
    # We assume:
    # - If it's Model's turn, this move is FROM the model (via frontend). -> Store for RL.
    # - If it's User's turn, this move is FROM the user. -> Train Supervised.
    is_model_turn = (board.turn == chess.WHITE and model_as_white) or \
                    (board.turn == chess.BLACK and not model_as_white)
    
    if is_model_turn:
        # Model played this move. Store for RL.
        try:
            if move_str in move_to_id:
                game_trajectory.append((board_to_tensor(board), move_to_id[move_str]))
        except Exception:
            pass
    else:
        # User played this move. Imitate it!
        train_step_supervised(board, move)

    board.push(move)
    
    if board.is_game_over():
        train_step_rl(board.result())
        
    return await get_state()


@app.post("/api/undo")
async def api_undo():
    if not board.move_stack:
        raise HTTPException(status_code=400, detail="No moves to undo")
    board.pop()
    return await get_state()


@app.post("/api/goto")
async def api_goto(payload: dict):
    ply = payload.get("ply")
    if ply is None or not isinstance(ply, int) or ply < 0:
        raise HTTPException(status_code=400, detail="Invalid ply")
    moves = list(board.move_stack)
    if ply > len(moves):
        raise HTTPException(status_code=400, detail="Ply exceeds move stack")
    reset_game()
    for mv in moves[:ply]:
        board.push(mv)
    return await get_state()


@app.post("/api/engine")
async def api_engine(payload: dict):
    global engine_mode, search_depth, search_time, model_temperature, model_epsilon
    mode = payload.get("mode", engine_mode)
    depth = payload.get("depth", search_depth)
    time_limit = payload.get("time", search_time)
    temp = payload.get("temperature", model_temperature)
    eps = payload.get("epsilon", model_epsilon)

    if mode not in {"stockfish", "model"}:
        raise HTTPException(status_code=400, detail="Invalid engine mode")
    try:
        depth_val = int(depth) if depth is not None else search_depth
    except ValueError:
        depth_val = search_depth
    search_depth = max(1, depth_val)
    search_time = float(time_limit) if time_limit is not None else None
    
    try:
        model_temperature = float(temp)
        model_epsilon = float(eps)
    except ValueError:
        pass

    engine_mode = mode
    return await get_state()


def load_policy(path: Path) -> None:
    global policy_model, move_to_id, id_to_move, policy_device
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Model not found at {path}")
    try:
        torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
    except Exception:
        pass
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    classes = ckpt.get("classes")
    if classes is None:
        # fallback for RL checkpoints without classes
        id_to_move = []
        move_to_id = {}
    else:
        id_to_move = list(classes)
        move_to_id = {m: i for i, m in enumerate(id_to_move)}
    arch = ckpt.get("arch", "mlp")
    hidden = ckpt.get("hidden", 768)
    channels = ckpt.get("channels", 128)
    blocks = ckpt.get("blocks", 4)
    state = ckpt.get("model_state", {})
    if any(k.startswith("net.") for k in state.keys()):
        new_state = OrderedDict()
        for k, v in state.items():
            new_key = k.replace("net.", "")
            new_state[new_key] = v
        state = new_state
        if "0.weight" in state:
            hidden = state["0.weight"].shape[0]
    # support RL conv policy
    if arch == "pv_conv":
        model = PolicyValueNet(channels=channels)
    elif arch == "mlp":
        n_classes = 4672 if not id_to_move else len(id_to_move)
        model = nn.Sequential(
            nn.Linear(13 * 64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported arch {arch}")
    model.load_state_dict(state, strict=False)
    policy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(policy_device)
    model.eval()
    policy_model = model
    global policy_optimizer
    policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-4)


# Exploration Parameters
model_temperature: float = 0.5
model_epsilon: float = 0.05

def policy_move() -> Optional[str]:
    if policy_model is None or not move_to_id:
        return None
    
    # Epsilon-Greedy: Random move
    legal = list(board.legal_moves)
    if not legal:
        return None
    if random.random() < model_epsilon:
        return random.choice(legal).uci()

    tensor = board_to_tensor(board).to(policy_device).unsqueeze(0).float()
    with torch.no_grad():
        logits = policy_model(tensor).squeeze(0)
    
    mask = torch.full_like(logits, float("-inf"))
    for mv in legal:
        u = mv.uci()
        idx = move_to_id.get(u)
        if idx is not None:
            mask[idx] = logits[idx]
    
    # Temperature Sampling
    if model_temperature > 0:
        probs = torch.softmax(mask / model_temperature, dim=0)
        try:
            idx = torch.multinomial(probs, 1).item()
        except RuntimeError:
            # Fallback if probs sum to 0 or other issue
            idx = torch.argmax(mask).item()
    else:
        # Greedy (Argmax)
        idx = torch.argmax(mask).item()
        
    return id_to_move[idx] if idx < len(id_to_move) else None


def policy_move_for_board(b: chess.Board) -> Optional[str]:
    if policy_model is None:
        return None
        
    # Epsilon-Greedy: Random move
    legal = list(b.legal_moves)
    if not legal:
        return None
    if random.random() < model_epsilon:
        return random.choice(legal).uci()

    tensor = board_to_tensor(b).to(policy_device).unsqueeze(0).float()
    with torch.no_grad():
        logits = policy_model(tensor).squeeze(0)
    
    # if we have label encoder, mask by uci; else assume logits over legal moves
    if move_to_id:
        mask = torch.full_like(logits, float("-inf"))
        for mv in legal:
            u = mv.uci()
            idx = move_to_id.get(u)
            if idx is not None:
                mask[idx] = logits[idx]
        
        # Temperature Sampling
        if model_temperature > 0:
            probs = torch.softmax(mask / model_temperature, dim=0)
            try:
                idx = torch.multinomial(probs, 1).item()
            except RuntimeError:
                idx = torch.argmax(mask).item()
        else:
            idx = torch.argmax(mask).item()
            
        return id_to_move[idx] if idx < len(id_to_move) else None
    else:
        # Fallback for models without class map (not expected for this project)
        probs = torch.softmax(logits[: len(legal)], dim=0)
        idx = torch.argmax(probs).item()
        return legal[idx].uci()


@app.get("/api/export_pgn")
async def api_export_pgn():
    game = chess.pgn.Game()
    temp_board = game.board()
    node = game
    for mv in board.move_stack:
        node = node.add_variation(mv)
        temp_board.push(mv)
    return {"pgn": str(game)}


@app.post("/api/import_pgn")
async def api_import_pgn(payload: dict):
    pgn_text = payload.get("pgn")
    if not pgn_text:
        raise HTTPException(status_code=400, detail="Missing PGN")
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise HTTPException(status_code=400, detail="Could not parse PGN")
    reset_game()
    for mv in game.mainline_moves():
        board.push(mv)
    return await get_state()


@app.post("/api/load_model")
async def api_load_model(payload: dict):
    global current_model_path
    path = payload.get("path", "checkpoints/supervised_policy.pt")
    try:
        load_policy(Path(path))
        current_model_path = Path(path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {e}")
    return {"status": "ok", "path": str(path), "classes": len(id_to_move)}


# Opponent State for Self-Play
opponent_model: Optional[nn.Module] = None
opponent_type: str = "stockfish"  # "stockfish", "self", "frozen"

@app.post("/api/load_opponent")
async def api_load_opponent(payload: dict):
    path = payload.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="Missing path")
    try:
        load_opponent_policy(Path(path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load opponent: {e}")
    return {"status": "ok", "path": str(path)}

def load_opponent_policy(path: Path) -> None:
    global opponent_model
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Model not found at {path}")
    try:
        torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
    except Exception:
        pass
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # We assume opponent uses same architecture/classes as main model for now
    # or at least compatible output if we just use it for inference
    arch = ckpt.get("arch", "mlp")
    hidden = ckpt.get("hidden", 768)
    channels = ckpt.get("channels", 128)
    state = ckpt.get("model_state", {})
    
    if any(k.startswith("net.") for k in state.keys()):
        new_state = OrderedDict()
        for k, v in state.items():
            new_key = k.replace("net.", "")
            new_state[new_key] = v
        state = new_state
        if "0.weight" in state:
            hidden = state["0.weight"].shape[0]

    if arch == "pv_conv":
        model = PolicyValueNet(channels=channels)
    elif arch == "mlp":
        # We need n_classes. If main model loaded, use its n_classes?
        # Or infer from checkpoint?
        # Let's assume same classes as main model for simplicity
        n_classes = 4672 if not id_to_move else len(id_to_move)
        model = nn.Sequential(
            nn.Linear(13 * 64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported arch {arch}")
    
    model.load_state_dict(state, strict=False)
    model.to(policy_device)
    model.eval()
    opponent_model = model


def log_eval(entry: dict) -> None:
    eval_history_path.parent.mkdir(parents=True, exist_ok=True)
    hist = []
    if eval_history_path.exists():
        try:
            hist = json.loads(eval_history_path.read_text())
        except Exception:
            hist = []
    hist.append(entry)
    eval_history_path.write_text(json.dumps(hist, indent=2))


def play_game_model_vs_stockfish(max_plies: int = 80) -> str:
    if engine is None:
        raise RuntimeError("Stockfish engine not available")
    if policy_model is None:
        raise RuntimeError("Policy model not loaded")
    b = chess.Board()
    result = None
    ply = 0
    while not b.is_game_over() and ply < max_plies:
        if b.turn == chess.WHITE:
            mv = policy_move_for_board(b)
            if mv is None:
                break
            b.push_uci(mv)
        else:
            limit = chess.engine.Limit(depth=search_depth, time=search_time)
            info = engine.play(b, limit=limit)
            if info.move is None:
                break
            b.push(info.move)
        ply += 1
    if b.is_game_over():
        result = b.result()
    else:
        result = "1/2-1/2"
    return result


@app.post("/api/eval")
async def api_eval(payload: dict):
    games = int(payload.get("games", 4))
    max_plies = int(payload.get("max_plies", 80))
    if engine is None:
        raise HTTPException(status_code=400, detail="Stockfish not available")
    if policy_model is None:
        raise HTTPException(status_code=400, detail="Policy model not loaded")
    w = d = l = 0
    for _ in range(games):
        res = play_game_model_vs_stockfish(max_plies=max_plies)
        if res == "1-0":
            w += 1
        elif res == "0-1":
            l += 1
        else:
            d += 1
    entry = {"games": games, "win": w, "draw": d, "loss": l, "ts": datetime.utcnow().isoformat()}
    log_eval(entry)
    return entry


@app.get("/api/eval_history")
async def api_eval_history():
    if not eval_history_path.exists():
        return {"history": [], "totals": {"win": 0, "draw": 0, "loss": 0}}
    try:
        hist = json.loads(eval_history_path.read_text())
    except Exception:
        hist = []
    totals = {"win": 0, "draw": 0, "loss": 0}
    for h in hist:
        totals["win"] += h.get("win", 1 if h.get("result") == "1-0" else 0)
        totals["loss"] += h.get("loss", 1 if h.get("result") == "0-1" else 0)
        totals["draw"] += h.get("draw", 1 if h.get("result") not in {"1-0", "0-1"} else 0)
    return {"history": hist, "totals": totals}


def match_loop_sync(max_plies: int = 120, delay: float = 0.0):
    global match_running, match_board, model_as_white, policy_model, policy_optimizer, opponent_model, opponent_type
    while match_running:
        match_board = chess.Board()
        ply = 0
        stockfish_samples: List[Tuple[torch.Tensor, int]] = []
        model_samples: List[Tuple[torch.Tensor, int]] = []
        
        # For "self" play (Current vs Current), we can collect data for BOTH sides.
        # For "frozen" play (Current vs Frozen), we only collect for Current.
        # For "stockfish" play, we collect for Current (RL) and Stockfish (Imitation).
        
        while match_running and match_board and not match_board.is_game_over() and ply < max_plies:
            is_white = match_board.turn == chess.WHITE
            
            # Determine who is playing this turn
            # model_as_white determines if "Current Model" plays White.
            
            is_current_model_turn = (is_white and model_as_white) or (not is_white and not model_as_white)
            
            move_to_make = None
            is_training_move = False # Whether to store this move for RL update
            
            if is_current_model_turn:
                # Current Model plays
                move_to_make = policy_move_for_board(match_board)
                is_training_move = True
            else:
                # Opponent plays
                if opponent_type == "stockfish":
                    limit = chess.engine.Limit(depth=search_depth, time=search_time)
                    info = engine.play(match_board, limit=limit)
                    if info.move:
                        move_to_make = info.move.uci()
                        # Imitation learning from Stockfish
                        sf_tensor = board_to_tensor(match_board)
                        if move_to_make in move_to_id:
                            stockfish_samples.append((sf_tensor, move_to_id[move_to_make]))
                
                elif opponent_type == "self":
                    # Current Model plays as opponent too!
                    move_to_make = policy_move_for_board(match_board)
                    is_training_move = True # We learn from both sides in self-play
                
                elif opponent_type == "frozen":
                    # Frozen Model plays
                    if opponent_model is not None:
                        # Use opponent_model to pick move
                        # We need a helper for opponent move, similar to policy_move_for_board but using opponent_model
                        # For now, let's inline or reuse logic?
                        # Let's reuse logic but swap model temporarily? No, thread unsafe.
                        # Let's just copy-paste logic for now or make helper.
                        # Inline logic for opponent_model:
                        tensor = board_to_tensor(match_board).to(policy_device).unsqueeze(0).float()
                        with torch.no_grad():
                            logits = opponent_model(tensor).squeeze(0)
                        legal = list(match_board.legal_moves)
                        if legal:
                            if move_to_id:
                                mask = torch.full_like(logits, float("-inf"))
                                for mv in legal:
                                    u = mv.uci()
                                    idx = move_to_id.get(u)
                                    if idx is not None:
                                        mask[idx] = logits[idx]
                                # Use same temp/epsilon for opponent? Or deterministic?
                                # Usually deterministic or low temp for frozen opponent is good to test against "best"
                                # But diversity is good too. Let's use same params for now.
                                if model_temperature > 0:
                                    probs = torch.softmax(mask / model_temperature, dim=0)
                                    try:
                                        idx = torch.multinomial(probs, 1).item()
                                    except RuntimeError:
                                        idx = torch.argmax(mask).item()
                                else:
                                    idx = torch.argmax(mask).item()
                                move_to_make = id_to_move[idx] if idx < len(id_to_move) else None
                            else:
                                probs = torch.softmax(logits[: len(legal)], dim=0)
                                idx = torch.argmax(probs).item()
                                move_to_make = legal[idx].uci()

            if move_to_make is None:
                break
                
            # Store sample if it's a training move
            if is_training_move:
                try:
                    idx = move_to_id[move_to_make]
                    # Store (State, Action, Side) so we can assign correct reward later
                    # Side: 1 for White, -1 for Black
                    side = 1 if is_white else -1
                    model_samples.append((board_to_tensor(match_board), idx, side))
                except Exception:
                    pass

            match_board.push_uci(move_to_make)
            ply += 1
            if delay > 0:
                time.sleep(delay)

        # Post-game updates
        
        # 1. Imitation (Stockfish)
        if stockfish_samples:
            try:
                for s in stockfish_samples:
                    rl_buffer.add((s[0], torch.tensor(s[1], dtype=torch.long), 1.0))
            except Exception:
                pass
        
        # 2. RL (Self/Frozen/Stockfish)
        if model_samples:
            result = match_board.result() if match_board.is_game_over() else "1/2-1/2"
            
            # Determine game outcome for White
            if result == "1-0":
                white_reward = 1.0
            elif result == "0-1":
                white_reward = -1.0
            else:
                white_reward = 0.0
            
            if white_reward != 0.0: # Only learn from decisive games? Or draws too?
                # AlphaZero learns from draws (0). But PG usually needs +1/-1.
                # If reward is 0, gradient is 0. So no update.
                # Let's skip draws for now to encourage winning.
                try:
                    for s in model_samples:
                        # s = (tensor, action_idx, side)
                        # side is 1 (White) or -1 (Black)
                        # Reward for this move = white_reward * side
                        # Example: White wins (1.0). White move (side 1) -> 1.0 * 1 = 1.0 (Good)
                        # Example: White wins (1.0). Black move (side -1) -> 1.0 * -1 = -1.0 (Bad)
                        # Example: Black wins (-1.0). White move (side 1) -> -1.0 * 1 = -1.0 (Bad)
                        # Example: Black wins (-1.0). Black move (side -1) -> -1.0 * -1 = 1.0 (Good)
                        r = white_reward * s[2]
                        rl_buffer.add((s[0], torch.tensor(s[1], dtype=torch.long), r))
                except Exception:
                    pass

        # log result
        if match_board.is_game_over():
            res = match_board.result()
        else:
            res = "1/2-1/2"
        log_eval({"ts": datetime.utcnow().isoformat(), "result": res, "opponent": opponent_type})
        
        # toggle starting color for next game
        model_as_white = not model_as_white
    match_running = False


@app.post("/api/start_match")
async def api_start_match(payload: dict):
    global match_running, match_board, match_task, engine, opponent_type
    
    opp = payload.get("opponent", "stockfish")
    if opp not in {"stockfish", "self", "frozen"}:
        raise HTTPException(status_code=400, detail="Invalid opponent type")
    opponent_type = opp

    if opponent_type == "stockfish":
        if engine is None:
            eng = maybe_load_engine()
            if eng is None:
                raise HTTPException(
                    status_code=400,
                    detail="Stockfish not available. Build a binary at third_party/stockfish/src/stockfish or install stockfish on PATH (/usr/games/stockfish).",
                )
            engine = eng
    elif opponent_type == "frozen":
        if opponent_model is None:
             raise HTTPException(status_code=400, detail="Opponent model not loaded. Load it first.")

    if policy_model is None:
        raise HTTPException(status_code=400, detail="Policy model not loaded")
    if match_running:
        return {"status": "already_running"}
    match_board = chess.Board()
    match_running = True
    max_plies = int(payload.get("max_plies", 120))
    delay = float(payload.get("delay", 1.0))
    match_task = asyncio.get_event_loop().run_in_executor(None, functools.partial(match_loop_sync, max_plies=max_plies, delay=delay))
    return {"status": "started"}


@app.post("/api/stop_match")
async def api_stop_match():
    global match_running, match_task
    match_running = False
    match_task = None
    return {"status": "stopped"}


@app.get("/api/match_state")
async def api_match_state():
    if match_board is None:
        return {"running": False}
    return {
        "running": match_running,
        "fen": match_board.fen(),
        "move_stack": [m.uci() for m in match_board.move_stack],
        "result": match_board.result() if match_board.is_game_over() else None,
        "model_as_white": model_as_white,
    }


# Serve static assets (index.html + any JS/CSS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
