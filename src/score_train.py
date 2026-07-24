import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader

from score_head import ScorePredictorClsSequential
from score_dataset import SequentialClsDataset
from score_eval_cls_sequential import run_autoregressive_eval

TRAIN_GAME_IDS = ["1", "2", "4", "5"]
EVAL_GAME_ID = "3"
VAL_CHECK_EVERY = 5
CHUNK_LEN = 256
MAX_FRAMES_AHEAD = 30 


def build_dataset(gid):
    cache_path = f"data/dino_cache/game_{gid}/cache.pt"
    event_json = f"data/OpenTT/annotations/train/game_{gid}.json"
    csv_path = f"runs/20260704-1040_baseline/game_{gid}_predictions.csv"
    print(f"Loading game_{gid} ...")
    return SequentialClsDataset(cache_path, event_json, csv_path,
                                 chunk_len=CHUNK_LEN, max_frames_ahead=MAX_FRAMES_AHEAD)


def run_epoch(model, datasets, optimizer, ce_loss, device, train):
    model.train() if train else model.eval()
    total_loss, n_chunks = 0.0, 0

    for ds in datasets:
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        hidden, score = None, None

        for batch in loader:
            if bool(batch["is_run_start"].item()):
                hidden, score = None, None

            feats = batch["features"].to(device)
            event_type = batch["event_type"].to(device)
            score_in = score if score is not None else batch["score_state"].to(device)

            with torch.set_grad_enabled(train):
                type_logits, hidden, score = model(feats, score_in, hidden=hidden)
                loss = ce_loss(type_logits.reshape(-1, 3), event_type.reshape(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()
            n_chunks += 1

    return total_loss / max(n_chunks, 1)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    print(f"Train games: {TRAIN_GAME_IDS} | Eval game: {EVAL_GAME_ID}")

    train_datasets = [build_dataset(gid) for gid in TRAIN_GAME_IDS]
    eval_datasets = [build_dataset(EVAL_GAME_ID)]

    type_counts = Counter()
    for ds in train_datasets:
        for i in range(len(ds)):
            for c in ds[i]["event_type"].tolist():
                type_counts[c] += 1
    total = sum(type_counts.values())
    w0 = total / (3 * type_counts.get(0, 1))
    w1 = total / (3 * type_counts.get(1, 1))
    w2 = total / (3 * type_counts.get(2, 1))
    class_weights = torch.tensor([w0, w1, w2]).to(device)
    print(f"Event type distribution: {dict(type_counts)}")
    print(f"Class weights: [BG={w0:.1f}, Left={w1:.1f}, Right={w2:.1f}]")

    feature_dim = train_datasets[0].feature_dim
    model = ScorePredictorClsSequential(feature_dim=feature_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    print("\nStarting sequential classification (TBPTT) training...")
    best_macro_f1 = 0.0

    for epoch in range(60):
        train_loss = run_epoch(model, train_datasets, optimizer, ce_loss, device, train=True)
        scheduler.step()

        if (epoch + 1) % VAL_CHECK_EVERY == 0 or epoch == 59:
            val_loss = run_epoch(model, eval_datasets, optimizer, ce_loss, device, train=False)
            print(f"Epoch [{epoch+1}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            _, _, macro_f1 = run_autoregressive_eval(model, eval_datasets, device, verbose=True)

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                torch.save(model.state_dict(), "score_predictor_cls_sequential.pth")
                print(f"  -> new best (macro F1={macro_f1:.3f}), saved")
        else:
            print(f"Epoch [{epoch+1}] | Train Loss: {train_loss:.4f}")

    print(f"\nBest macro F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    train()