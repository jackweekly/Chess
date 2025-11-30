import asyncio
import json
import os
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import contextlib

import chess
import chess.engine
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.models.alphazero import AlphaZeroNet
from src.rl.encoders import get_input_tensor
from src.rl.self_play_mcts import MCTS, load_move_encoder

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Chess Assistant")

# --- Global State ---
board = chess.Board()
engine: Optional[chess.engine.SimpleEngine] = None
engine_name = "none"
engine_mode = "stockfish"  # "stockfish" or "model"
search_depth = 12
search_time = None

# RL / Model State
policy_model: Optional[nn.Module] = None
mcts_instance: Optional[MCTS] = None
policy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
idx_to_move: List[str] = []
move_to_idx: Dict[str, int] = {}
eval_history: List[Dict[str, str]] = []
current_match: Dict[str, object] = {
    "running": False,
    "result": None,
    "model_as_white": True,
}


def maybe_load_engine() -> Optional[chess.engine.SimpleEngine]:
    """Tries to load Stockfish from common paths."""
    candidates = [
        Path("third_party/stockfish/src/stockfish"),
        Path("/usr/games/stockfish"),
        Path("/usr/local/bin/stockfish"),
    ]
    for cand in candidates:
        if cand.exists():
            try:
                return chess.engine.SimpleEngine.popen_uci(str(cand))
            except Exception:
                continue
    try:
        return chess.engine.SimpleEngine.popen_uci("stockfish")
    except Exception:
        return None


# --- Core Logic ---

def run_mcts_move(b: chess.Board, simulations: int = 100) -> Optional[str]:
    """Runs MCTS and returns the best move UCI."""
    if mcts_instance is None:
        return None
    return mcts_instance.get_best_move(b, sims=simulations)


async def suggest_move() -> dict:
    if engine_mode == "model":
        if policy_model is None:
            return {"move": None, "source": "model (not loaded)"}
        best = run_mcts_move(board, simulations=100)
        tensor = get_input_tensor(board).to(policy_device).unsqueeze(0)
        with torch.no_grad():
            _, v = policy_model(tensor)
        return {"move": best, "score": float(v.item()), "source": "AlphaZero (MCTS)"}

    if engine is None:
        return {"source": "none"}

    try:
        limit = chess.engine.Limit(depth=search_depth, time=0.1 if search_time is None else search_time)
        info = await engine.play(board, limit=limit, info=chess.engine.INFO_ALL)
        return {
            "move": info.move.uci() if info.move else None,
            "score": info.info.get("score").white().score() if info.info.get("score") else None,
            "source": f"Stockfish {search_depth}",
        }
    except Exception:
        return {"source": "error"}


@app.post("/api/move")
async def api_move(payload: dict):
    """
    The UI sends a move request. If payload has "auto": true, the bot moves.
    """
    auto = payload.get("auto", False)
    if auto:
        if engine_mode == "model":
            best = run_mcts_move(board, simulations=100)
            if best:
                board.push_uci(best)
            return await get_state()
        if engine is None:
            raise HTTPException(400, "Engine not available")
        try:
            limit = chess.engine.Limit(depth=search_depth, time=0.1 if search_time is None else search_time)
            info = await engine.play(board, limit=limit)
            if info.move:
                board.push(info.move)
        except Exception as e:
            raise HTTPException(500, f"Engine error: {e}")
        return await get_state()

    move_uci = payload.get("move")
    if not move_uci:
        raise HTTPException(400, "Missing move")
    try:
        move = chess.Move.from_uci(move_uci)
    except Exception:
        raise HTTPException(400, "Invalid move format")
    if move not in board.legal_moves:
        raise HTTPException(400, "Illegal move")
    board.push(move)
    return await get_state()


@app.get("/api/state")
async def get_state():
    suggestion = await suggest_move()
    san_stack = []
    temp = chess.Board()
    for mv in board.move_stack:
        san_stack.append(temp.san(mv))
        temp.push(mv)
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "move_stack": [m.uci() for m in board.move_stack],
        "san_stack": san_stack,
        "legal_moves": [m.uci() for m in board.legal_moves],
        "is_game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "suggestion": suggestion,
        "engine": {
            "mode": engine_mode,
            "name": engine_name,
            "depth": search_depth,
        },
    }


@app.post("/api/reset")
async def api_reset(payload: dict):
    fen = payload.get("fen")
    try:
        global board
        board = chess.Board(fen) if fen else chess.Board()
    except Exception as e:
        raise HTTPException(400, f"Invalid FEN: {e}")
    return await get_state()

@app.post("/api/undo")
async def api_undo():
    if board.move_stack:
        board.pop()
    return await get_state()

@app.post("/api/goto")
async def api_goto(payload: dict):
    ply = int(payload.get("ply", 0))
    while len(board.move_stack) > ply:
        board.pop()
    return await get_state()

# --- Evaluation / Match placeholders ---


@app.post("/api/eval")
async def api_eval(payload: dict):
    games = payload.get("games", 0)
    eval_history.append({"ts": datetime.utcnow().isoformat(), "result": f"queued {games} games"})
    return {"status": "queued", "games": games}


@app.get("/api/eval_history")
async def api_eval_history():
    totals = {"win": 0, "draw": 0, "loss": 0}
    return {"history": eval_history, "totals": totals}


@app.post("/api/start_match")
async def api_start_match(payload: dict):
    current_match["running"] = True
    current_match["result"] = None
    current_match["model_as_white"] = True
    return {"status": "started"}


@app.post("/api/stop_match")
async def api_stop_match():
    current_match["running"] = False
    return {"status": "stopped"}


@app.get("/api/match_state")
async def api_match_state():
    return {
        "fen": board.fen(),
        "move_stack": [m.uci() for m in board.move_stack],
        "running": current_match["running"],
        "result": current_match["result"],
        "model_as_white": current_match["model_as_white"],
    }


@app.post("/api/load_model")
async def api_load_model(payload: dict):
    path = Path(payload.get("path", "checkpoints/rl_latest.pt"))
    moves_path = Path(payload.get("moves_path", "data/processed/supervised/label_encoder_classes.npy"))
    if not path.exists():
        raise HTTPException(400, "Model not found")

    global policy_model, mcts_instance, idx_to_move, move_to_idx
    ckpt = torch.load(path, map_location=policy_device)
    if "moves" in ckpt:
        idx_to_move = [str(x) for x in ckpt["moves"]]
        move_to_idx = {uci: i for i, uci in enumerate(idx_to_move)}
    else:
        if not moves_path.exists():
            raise HTTPException(400, "Move list not found and checkpoint missing embedded moves")
        idx_to_move, move_to_idx = load_move_encoder(moves_path)
    action_size = len(idx_to_move)
    model = AlphaZeroNet(
        channels=ckpt.get("channels", 128),
        blocks=ckpt.get("blocks", 10),
        n_classes=action_size,
        input_channels=119,
    )
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.to(policy_device)
    model.eval()
    policy_model = model
    mcts_instance = MCTS(
        policy_model,
        device=policy_device,
        sims=100,
        move_to_idx=move_to_idx,
        idx_to_move=idx_to_move,
        batch_size=32,
    )
    return {"status": "loaded", "path": str(path), "moves": len(idx_to_move)}


@app.on_event("startup")
async def startup_event() -> None:
    global engine, engine_name
    engine = maybe_load_engine()
    if engine:
        with contextlib.suppress(Exception):
            engine_name = engine.id.get("name", "engine")
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global engine
    if engine is not None:
        with contextlib.suppress(Exception):
            engine.quit()
        engine = None


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))
