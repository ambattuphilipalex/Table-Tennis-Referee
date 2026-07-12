import torch
from score_head import ScorePredictor
from score_dataset import SequenceScoreDataset
from score_eval_utils import evaluate_model, load_gt_events
from score_train import EVAL_GAME_ID

THRESHOLDS = [0.34, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_path = f"data/dino_cache/game_{EVAL_GAME_ID}/cache.pt"
    event_json = f"data/OpenTT/annotations/train/game_{EVAL_GAME_ID}.json"
    csv_path = f"runs/20260704-1040_baseline/game_{EVAL_GAME_ID}_predictions.csv"

    gt_left, gt_right = load_gt_events(event_json)
    dataset = SequenceScoreDataset(cache_path, event_json, csv_path)
    inferred_input_dim = dataset[0][0].shape[-1]

    model = ScorePredictor(input_dim=inferred_input_dim).to(device)
    model.load_state_dict(torch.load("score_predictor.pth", map_location=device))
    model.eval()

    print(f"{'threshold':>10} {'recall':>8} {'precision':>10} {'f1':>7} {'pred (L/R)':>12}")
    best = None
    for t in THRESHOLDS:
        m = evaluate_model(model, dataset, gt_left, gt_right, device,
                           verbose=False, min_confidence=t)
        print(f"{t:>10.2f} {m['recall']*100:>7.1f}% {m['precision']*100:>9.1f}% "
              f"{m['f1']:>7.3f} {m['left_points']:>5d}L/{m['right_points']:<5d}R")
        if best is None or m["f1"] > best[1]["f1"]:
            best = (t, m)

    t, m = best
    print(f"\nBest threshold: {t:.2f} -> f1={m['f1']:.3f}, "
          f"recall={m['recall']*100:.1f}%, precision={m['precision']*100:.1f}%")
    print("Use this threshold's value as min_confidence in score_predict.py's "
          "evaluate_model() call once you're happy with the trade-off.")


if __name__ == "__main__":
    main()