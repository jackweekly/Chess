import json
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AlphaZero Dashboard")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
EVAL_HISTORY_PATH = Path("data/eval/history.json")
match_state = {"running": False, "result": None}


@app.get("/health.ico", include_in_schema=False)
async def health_check():
    return Response(status_code=204)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Dashboard not found. Check src/web/static/index.html</h1>")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/metrics")
async def get_metrics():
    data = {"labels": [], "win_rates": [], "loss": []}
    if EVAL_HISTORY_PATH.exists():
        try:
            history = json.loads(EVAL_HISTORY_PATH.read_text())
            for entry in history[-100:]:
                label = entry.get("ts", "??").split("T")[-1][:5]
                data["labels"].append(label)
                games = entry.get("games", 1) or 1
                wins = entry.get("win", 0)
                data["win_rates"].append(round(wins / games, 2))
                data["loss"].append(entry.get("loss_val", 0))
        except Exception as e:
            print(f"Error reading metrics: {e}")
    return data


@app.post("/api/start_match")
async def start_match():
    match_state["running"] = True
    match_state["result"] = None
    return {"status": "started"}


@app.post("/api/stop_match")
async def stop_match():
    match_state["running"] = False
    return {"status": "stopped"}


@app.get("/api/match_state")
async def match_state_api():
    return match_state


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
