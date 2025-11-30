.PHONY: install download prep train self-play stockfish stop run-web

PYTHON ?= python3
KAGGLE_CONFIG_DIR := $(PWD)
RAW_DIR := data/raw
PROC_DIR := data/processed/supervised
PGN_FILE ?= $(RAW_DIR)/chess_games.csv
MAX_GAMES ?= 0          # 0 = all games
MAX_MOVES ?= 0          # 0 = full games
EPOCHS ?= 5
BATCH_SIZE ?= 128
GPUS ?= 1
SHARD_SIZE ?= 250000
VAL_FRAC ?= 0.1
CSV_ENGINE ?= pyarrow
CSV_CHUNKSIZE ?= 1000
CSV_THREADS ?= 16
CSV_BLOCK_SIZE ?= 8388608
NUM_WORKERS ?= 16
READ_BATCH ?= 16384
MAX_TRAIN_BATCHES ?= 2000
MAX_VAL_BATCHES ?= 200
TRIALS ?= 4
TUNE_MAX_TRAIN_BATCHES ?= 50
TUNE_MAX_VAL_BATCHES ?= 20
TUNE_MAX_TRAIN_FILES ?= 1
TUNE_MAX_VAL_FILES ?= 1
TRAIN_EXTRA ?=
TUNE_EXTRA ?=
WEB_PORT ?= 8000
STOCKFISH_REPO := https://github.com/official-stockfish/Stockfish.git
STOCKFISH_DIR := third_party/stockfish

install:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "uv not found; installing via pip..."; \
		$(PYTHON) -m pip install --upgrade pip && $(PYTHON) -m pip install uv; \
	else \
		echo "uv found; using existing installation."; \
	fi
	uv pip install -r requirements.txt

download: stockfish
	KAGGLE_CONFIG_DIR=$(KAGGLE_CONFIG_DIR) $(PYTHON) src/data/download_dataset.py --output-dir $(RAW_DIR)

stockfish:
	@mkdir -p $(dir $(STOCKFISH_DIR))
	@if [ -d "$(STOCKFISH_DIR)/.git" ]; then \
		echo "Checking Stockfish for updates..."; \
		git -C $(STOCKFISH_DIR) fetch origin; \
		if git -C $(STOCKFISH_DIR) status -sb | grep -q "\[behind"; then \
			echo "Updating Stockfish..."; \
			git -C $(STOCKFISH_DIR) pull --ff-only; \
		else \
			echo "Stockfish already up to date."; \
		fi; \
	else \
		echo "Cloning Stockfish..."; \
		git clone $(STOCKFISH_REPO) $(STOCKFISH_DIR); \
	fi
	@echo "Building Stockfish..."
	@$(MAKE) -C $(STOCKFISH_DIR)/src build

prep:
	PYARROW_NUM_THREADS=$(CSV_THREADS) OMP_NUM_THREADS=$(CSV_THREADS) NUMEXPR_MAX_THREADS=$(CSV_THREADS) $(PYTHON) src/data/prepare_supervised_dataset.py --pgn $(PGN_FILE) --output-dir $(PROC_DIR) --max-games $(MAX_GAMES) --max-moves $(MAX_MOVES) --val-frac $(VAL_FRAC) --shard-size $(SHARD_SIZE) --csv-engine $(CSV_ENGINE) --csv-chunksize $(CSV_CHUNKSIZE) --csv-threads $(CSV_THREADS) --csv-block-size $(CSV_BLOCK_SIZE) --num-workers $(NUM_WORKERS)

run:
	$(MAKE) prep PGN_FILE=data/raw/chess_games.csv SAN_COLUMN=AN MAX_GAMES=0 MAX_MOVES=0 SHARD_SIZE=200000 VAL_FRAC=0.1

train:
	@if [ $(GPUS) -gt 1 ]; then \
		torchrun --nproc_per_node=$(GPUS) src/models/supervised_baseline.py --data-dir $(PROC_DIR) --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --arch mlp --hidden 768 --read-batch-size $(READ_BATCH) --num-workers $(NUM_WORKERS) --max-train-batches $(MAX_TRAIN_BATCHES) --max-val-batches $(MAX_VAL_BATCHES) --lr 0.001134 --cudnn-benchmark --non-blocking $(TRAIN_EXTRA); \
	else \
		$(PYTHON) src/models/supervised_baseline.py --data-dir $(PROC_DIR) --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --arch mlp --hidden 768 --read-batch-size $(READ_BATCH) --num-workers $(NUM_WORKERS) --max-train-batches $(MAX_TRAIN_BATCHES) --max-val-batches $(MAX_VAL_BATCHES) --lr 0.001134 --cudnn-benchmark --non-blocking $(TRAIN_EXTRA); \
	fi

self-play:
	@if [ $(GPUS) -gt 1 ]; then \
		torchrun --nproc_per_node=$(GPUS) src/rl/self_play.py; \
	else \
		$(PYTHON) src/rl/self_play.py; \
	fi

stop:
	@echo "Stopping training/self-play processes..."
	@# Kill prepare_supervised_dataset.py and its children recursively
	@for pid in $$(pgrep -f "src/data/prepare_supervised_dataset.py"); do \
		echo "Killing prepare_supervised_dataset.py (PID $$pid) and children..."; \
		pkill -P $$pid 2>/dev/null || true; \
		kill $$pid 2>/dev/null || true; \
	done
	@pkill -f "src/models/supervised_baseline.py" 2>/dev/null || true
	@pkill -f "src/rl/self_play.py" 2>/dev/null || true
	@pkill -f "torchrun.*supervised_baseline.py" 2>/dev/null || true
	@pkill -f "torchrun.*self_play.py" 2>/dev/null || true
	@pkill -f "prepare_supervised_dataset.py" 2>/dev/null || true
	@pkill -f "python3 src/data/prepare_supervised_dataset.py" 2>/dev/null || true
	@pkill -f "python.*prepare_supervised_dataset.py" 2>/dev/null || true
	@pkill -9 -f "prepare_supervised_dataset.py" 2>/dev/null || true
	@pkill -9 -f "python.*prepare_supervised_dataset.py" 2>/dev/null || true
	@pkill -f "make run" 2>/dev/null || true
	@pkill -f "make prep" 2>/dev/null || true
	@pkill -f "tune_supervised.py" 2>/dev/null || true
	@pkill -9 -f "tune_supervised.py" 2>/dev/null || true
	@pkill -f "python3 src/models/tune_supervised.py" 2>/dev/null || true
	@pkill -9 -f "python3 src/models/tune_supervised.py" 2>/dev/null || true
	@pkill -f "python.*tune_supervised.py" 2>/dev/null || true
	@pkill -9 -f "python.*tune_supervised.py" 2>/dev/null || true
	@pkill -f "make tune" 2>/dev/null || true
	@pkill -9 -f "make tune" 2>/dev/null || true
	@pkill -f "torchrun.*tune_supervised.py" 2>/dev/null || true
	@pkill -9 -f "torchrun.*tune_supervised.py" 2>/dev/null || true
	@pkill -f "python3 .*supervised_baseline.py" 2>/dev/null || true
	@pkill -9 -f "python3 .*supervised_baseline.py" 2>/dev/null || true

run-web:
	$(PYTHON) -m uvicorn src.web.app:app --host 0.0.0.0 --port $(WEB_PORT)

tune:
	$(PYTHON) src/models/tune_supervised.py --data-dir $(PROC_DIR) --trials $(TRIALS) --val-batches 8 --max-train-batches $(TUNE_MAX_TRAIN_BATCHES) --max-val-batches $(TUNE_MAX_VAL_BATCHES) --max-train-files $(TUNE_MAX_TRAIN_FILES) --max-val-files $(TUNE_MAX_VAL_FILES) $(TUNE_EXTRA)
