import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_stage2 import evaluate_stage2, append_result
from score_head import ScorePredictorClsSequential
from score_dataset import SequentialClsDataset
from score_eval_utils import load_gt_events, check_scoreboard_change


@torch.no_grad()
def predict_and_evaluate(cache_path, event_json, csv_path,
                         model_path="score_predictor_cls_sequential.pth",
                         min_confidence=0.90, chunk_len=256, max_frames_ahead=30,
                         use_zone_filter=False, use_scoreboard_verification=False,
                         exp_id=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gt_left, gt_right = load_gt_events(event_json)
    print(f"Ground truth: {len(gt_left)} left, {len(gt_right)} right")

    dataset = SequentialClsDataset(cache_path, event_json, csv_path,
                                   chunk_len=chunk_len,
                                   max_frames_ahead=max_frames_ahead)

    model = ScorePredictorClsSequential(feature_dim=dataset.feature_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    hidden, score = None, None
    per_frame = []

    for batch in loader:
        if bool(batch["is_run_start"].item()):
            hidden, score = None, None
        feats = batch["features"].to(device)
        frame_numbers = batch["frame_numbers"][0].tolist()
        score_in = score if score is not None else batch["score_state"].to(device)

        type_logits, hidden, score = model(feats, score_in, hidden=hidden)
        probs = torch.softmax(type_logits, dim=-1)[0]
        pred_cls = torch.argmax(type_logits, dim=-1)[0]

        for i, real_frame in enumerate(frame_numbers):
            cls = pred_cls[i].item()
            conf = probs[i, cls].item()
            if use_scoreboard_verification and cls != 0:
                confirmed = check_scoreboard_change(dataset, real_frame, num_frames=150)
                conf = min(conf + 0.15, 1.0) if confirmed else max(conf - 0.15, 0.0)
            per_frame.append((int(real_frame), int(cls), float(conf)))

    metrics = evaluate_stage2(per_frame, gt_left, gt_right, dataset.coords_dict,
                              min_confidence=min_confidence,
                              use_zone_filter_at_predict=use_zone_filter,
                              max_frames_ahead=max_frames_ahead, verbose=True)
    if exp_id:
        append_result(exp_id, "cls", None, metrics,
                      {"model_path": model_path, "min_conf": min_confidence,
                       "zone_predict": use_zone_filter,
                       "scoreboard": use_scoreboard_verification,
                       "csv_path": csv_path},
                      game=Path(cache_path).parent.name)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="game_3")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="score_predictor_cls_sequential.pth")
    ap.add_argument("--csv-dir", default="runs/20260704-1040_baseline")
    # 0.90 swept on the TRAINING games only: cuts game_3 scoreline error 8.7 -> 6.0
    # +/- 1.0 at unchanged event F1 (0.794 +/- 0.024). See CHANGELOG.md.
    ap.add_argument("--min-conf", type=float, default=0.90)
    # Zone filter off: E2b measured it at prediction time as changing event F1 by
    # exactly 0.000 and detection count not at all, across all three seeds.
    ap.add_argument("--zone-filter", type=int, default=0)
    ap.add_argument("--scoreboard", action="store_true")
    ap.add_argument("--exp-id", default=None)
    a = ap.parse_args()
    predict_and_evaluate(
        f"data/dino_cache/{a.game}/cache.pt",
        f"data/OpenTT/annotations/{a.split}/{a.game}.json",
        f"{a.csv_dir}/{a.game}_predictions.csv",
        model_path=a.model, min_confidence=a.min_conf,
        use_zone_filter=bool(a.zone_filter),
        use_scoreboard_verification=a.scoreboard,
        exp_id=a.exp_id)
