.PHONY: install download prep train-supervised train-rl play clean stop stop-tensorboard stop-train-rl

PYTHON ?= python3
# --- Hardware & Compute ---
GPUS ?= 1
NUM_WORKERS ?= 16

# --- Model Architecture ("HUGE" Settings) ---
MODEL_ARCH := conv
MODEL_CHANNELS := 128
MODEL_BLOCKS := 10

# --- Training Hyperparams ---
BATCH_SIZE := 256
EPOCHS := 100
MCTS_SIMS := 800
BUFFER_CAP := 200000

# --- Web UI Configuration ---
WEB_PORT := 4444
TENSORBOARD_PORT ?= 6006

# --- Paths ---
RAW_DIR := data/raw
PROC_DIR := data/processed/supervised
CHECKPOINT_DIR := checkpoints
SUPERVISED_CKPT := $(CHECKPOINT_DIR)/supervised_policy.pt
RL_CKPT_DIR := $(CHECKPOINT_DIR)/rl
LOG_DIR := logs

# --- Assets ---
PIECE_SET := staunty
PIECE_LIST := bB bK bN bP bQ bR wB wK wN wP wQ wR

# --- Setup Commands ---

install:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Installing uv..."; \
		$(PYTHON) -m pip install uv; \
	fi
	uv pip install -r requirements.txt

# --- Phase 1: Supervised Learning (Pre-training) ---
train-supervised:
	@echo "Starting Supervised Pre-training (ResNet: $(MODEL_CHANNELS)ch, $(MODEL_BLOCKS)blk)..."
	@mkdir -p $(CHECKPOINT_DIR)
	@if [ $(GPUS) -gt 1 ]; then \
		torchrun --nproc_per_node=$(GPUS) src/models/supervised_baseline.py \
			--data-dir $(PROC_DIR) \
			--arch $(MODEL_ARCH) \
			--channels $(MODEL_CHANNELS) \
			--blocks $(MODEL_BLOCKS) \
			--batch-size $(BATCH_SIZE) \
			--epochs $(EPOCHS) \
			--save-dir $(CHECKPOINT_DIR) \
			--num-workers $(NUM_WORKERS); \
	else \
		$(PYTHON) src/models/supervised_baseline.py \
			--data-dir $(PROC_DIR) \
			--arch $(MODEL_ARCH) \
			--channels $(MODEL_CHANNELS) \
			--blocks $(MODEL_BLOCKS) \
			--batch-size $(BATCH_SIZE) \
			--epochs $(EPOCHS) \
			--save-dir $(CHECKPOINT_DIR) \
			--num-workers $(NUM_WORKERS); \
	fi

# --- Phase 2: Reinforcement Learning (Self-Play) ---
train-rl:
	@echo "Starting AlphaZero Loop (MCTS Sims: $(MCTS_SIMS))..."
	@mkdir -p $(RL_CKPT_DIR)
	@if [ $(GPUS) -gt 1 ]; then \
		PYTHONPATH=. torchrun --nproc_per_node=$(GPUS) -m src.rl.self_play_mcts \
			--channels $(MODEL_CHANNELS) \
			--blocks $(MODEL_BLOCKS) \
			--mcts-sims $(MCTS_SIMS) \
			--batch-size $(BATCH_SIZE) \
			--buffer-cap $(BUFFER_CAP) \
			--save-dir $(RL_CKPT_DIR); \
	else \
		PYTHONPATH=. $(PYTHON) -m src.rl.self_play_mcts \
			--channels $(MODEL_CHANNELS) \
			--blocks $(MODEL_BLOCKS) \
			--mcts-sims $(MCTS_SIMS) \
			--batch-size $(BATCH_SIZE) \
			--buffer-cap $(BUFFER_CAP) \
			--save-dir $(RL_CKPT_DIR); \
	fi

# --- Phase 3: Deployment / Visualization ---
play:
	$(PYTHON) -m uvicorn src.web.app:app --host 0.0.0.0 --port $(WEB_PORT) --reload

# --- Utilities ---

tensorboard:
	tensorboard --logdir $(LOG_DIR) --port $(TENSORBOARD_PORT)

clean:
	rm -rf __pycache__ .pytest_cache
	find . -name "*.pyc" -delete

stop:
	@echo "Stopping all training processes..."
	@pkill -f "self_play_mcts.py" || true
	@pkill -f "supervised_baseline.py" || true
	@pkill -f "uvicorn" || true
	@pkill -f "tensorboard" || true

stop-tensorboard:
	@echo "Stopping tensorboard..."
	@pkill -f "tensorboard" || true

stop-train-rl:
	@echo "Stopping train-rl (self_play_mcts.py)..."
	@pkill -f "self_play_mcts.py" || true

download-pieces:
	@echo "Downloading chess piece SVGs ($(PIECE_SET))..."
	@cd src/web/static/pieces && for p in $(PIECE_LIST); do \
		curl -sSL -o $${p}.svg https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/$(PIECE_SET)/$${p}.svg; \
	done
