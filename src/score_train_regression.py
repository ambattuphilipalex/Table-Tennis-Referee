import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_stage2 import append_result
from run_experiment import build_dataset, run_one

TRAIN_GAME_IDS = ["1", "2", "4", "5"]
EVAL_GAME_ID = "3"
VAL_CHECK_EVERY = 5
CHUNK_LEN = 256

CFG = {
    "arm": "reg",
    "train_games": [f"game_{g}" for g in TRAIN_GAME_IDS],
    "eval_games": [f"game_{EVAL_GAME_ID}"],
    "epochs": 60, "val_every": VAL_CHECK_EVERY,
    "min_conf": 0.65,
    "note": "baseline config pinned by score_train_regression.py",
}


def train(seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    print(f"Train games: {TRAIN_GAME_IDS} | Eval game: {EVAL_GAME_ID}")

    train_ds = [build_dataset(g, CFG) for g in CFG["train_games"]]
    eval_ds = [build_dataset(g, CFG) for g in CFG["eval_games"]]

    metrics, model = run_one(CFG, seed, train_ds, eval_ds, device)
    torch.save(model.state_dict(), "score_predictor_sequential.pth")
    print("saved score_predictor_sequential.pth")
    append_result("score_train_regression", "reg", seed, metrics, CFG,
                  game=f"game_{EVAL_GAME_ID}")
    return metrics


if __name__ == "__main__":
    train(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
