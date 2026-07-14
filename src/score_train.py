import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from score_head import ScorePredictor
from score_dataset import SequenceScoreDataset
from score_eval_utils import evaluate_model, load_gt_events

TRAIN_GAME_IDS = ["1", "2", "4", "5"]
EVAL_GAME_ID = "3"
VAL_CHECK_EVERY = 5
MAX_CLASS_WEIGHT = 100.0


def build_dataset(gid):
    cache_path = f"data/dino_cache/game_{gid}/cache.pt"
    event_json = f"data/OpenTT/annotations/train/game_{gid}.json"
    csv_path = f"runs/20260704-1040_baseline/game_{gid}_predictions.csv"
    return SequenceScoreDataset(cache_path, event_json, csv_path)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    print(f"Train games: {TRAIN_GAME_IDS} | Held-out eval game: {EVAL_GAME_ID}")

    dataset_list = []
    per_dataset_labels = [] 
    total_counts = Counter()
    for gid in TRAIN_GAME_IDS:
        ds = build_dataset(gid)
        labels = [ds.label_only(i) for i in range(len(ds))]
        counts = Counter(labels)
        total_counts.update(counts)
        dataset_list.append(ds)
        per_dataset_labels.append(labels)
        print(f"Loaded game_{gid}: {len(ds)} windows, label counts {dict(counts)}")

    print(f"\nOverall training label distribution: {dict(total_counts)}")

    n0 = total_counts.get(0, 1)
    n1 = max(total_counts.get(1, 1), 1)
    n2 = max(total_counts.get(2, 1), 1)
    w1 = min(n0 / n1, MAX_CLASS_WEIGHT)
    w2 = min(n0 / n2, MAX_CLASS_WEIGHT)
    class_weights = torch.tensor([1.0, w1, w2]).to(device)
    print(f"Computed class weights: [1.0, {w1:.1f}, {w2:.1f}]")

    full_dataset = ConcatDataset(dataset_list)
    all_labels = [lbl for labels in per_dataset_labels for lbl in labels]

    weight_lookup = {0: 1.0, 1: w1, 2: w2}
    sample_weights = [weight_lookup[l] for l in all_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    loader = DataLoader(full_dataset, batch_size=32, sampler=sampler,
                        num_workers=2, pin_memory=(device == "cuda"))

    inferred_input_dim = dataset_list[0][0][0].shape[-1]
    print(f"Inferred input_dim={inferred_input_dim}")
    model = ScorePredictor(input_dim=inferred_input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    eval_cache = f"data/dino_cache/game_{EVAL_GAME_ID}/cache.pt"
    eval_events = f"data/OpenTT/annotations/train/game_{EVAL_GAME_ID}.json"
    eval_csv = f"runs/20260704-1040_baseline/game_{EVAL_GAME_ID}_predictions.csv"
    eval_dataset = SequenceScoreDataset(eval_cache, eval_events, eval_csv)
    gt_left, gt_right = load_gt_events(eval_events)

    print("\nStarting training...")
    best_f1 = -1.0
    for epoch in range(60):
        model.train()
        total_loss = 0.0
        for tokens, labels in loader:
            tokens, labels = tokens.to(device).float(), labels.to(device)
            optimizer.zero_grad()
            logits = model(tokens)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch + 1}] | Avg Loss: {avg_loss:.4f}")

        if (epoch + 1) % VAL_CHECK_EVERY == 0 or epoch == 59:
            metrics = evaluate_model(model, eval_dataset, gt_left, gt_right, device, verbose=False, min_confidence=0.65)
            print(f"  val on game_{EVAL_GAME_ID}: recall={metrics['recall']*100:.1f}% "
                  f"precision={metrics['precision']*100:.1f}% f1={metrics['f1']:.3f} "
                  f"predicted={metrics['left_points']}L/{metrics['right_points']}R")
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save(model.state_dict(), "score_predictor.pth")
                print(f"  -> new best (f1={best_f1:.3f}), checkpoint saved")
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), f"score_predictor_epoch{epoch+1}.pth")

    if best_f1 < 0:
        torch.save(model.state_dict(), "score_predictor.pth")
    print(f"\nBest validation F1: {best_f1:.3f}. Model saved as score_predictor.pth")


if __name__ == "__main__":
    train()