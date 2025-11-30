## Chess RL Starter

Scripts to (1) download Kaggle chess PGNs, (2) build a small supervised move predictor, and (3) kick off a toy self-play RL loop.

### Setup
- Install deps (ideally in a venv): `pip install -r requirements.txt`
- Dataset download (assumes Kaggle API creds are configured): `python src/data/download_dataset.py --output-dir data/raw`
  - The Kaggle dump includes multiple PGN files; pick one for preprocessing.

### Build a supervised baseline
- Prepare a split; defaults are small, but you can stream the full CSV (supports PGN or SAN columns). Examples:  
  - Small sanity check: `python src/data/prepare_supervised_dataset.py --pgn data/raw/chess_games.csv --output-dir data/processed/supervised --max-games 200 --max-moves 60`  
  - Full dataset (streaming shards): `python src/data/prepare_supervised_dataset.py --pgn data/raw/chess_games.csv --output-dir data/processed/supervised --max-games 0 --max-moves 0 --shard-size 200000 --val-frac 0.1 --san-column AN`
- Train a simple MLP on the extracted (board, move) pairs:  
  `python src/models/supervised_baseline.py --data-dir data/processed/supervised --epochs 5`  
  - Multi-GPU: `torchrun --nproc_per_node=2 src/models/supervised_baseline.py --data-dir data/processed/supervised --epochs 5`
- The label encoder classes are saved alongside the splits for later inference or RL warm-starts.

### Toy reinforcement learning loop
- Run a minimal self-play REINFORCE example (random-strength policy to start):  
  `python src/rl/self_play.py`
- The loop samples legal moves, assigns terminal rewards, and updates the policy. Extend with a stronger policy head and MCTS for better play.

### Next steps
- Increase the supervised corpus and architecture size to improve move prediction before self-play.
- Add checkpointing, evaluation against stockfish, and temperature-based move sampling.
- Swap in a board encoder with convolutional or transformer layers and integrate MCTS (AlphaZero-style) for stronger learning.
