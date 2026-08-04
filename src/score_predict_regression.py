import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_stage2 import evaluate_stage2, append_result
from score_head_regression import ScorePredictorSequential
from score_dataset_regression import SequentialScoreDataset
from score_eval_utils import load_gt_events, verify_visual_score_change


@torch.no_grad()
def predict_regression(cache_path, event_json, csv_path,
                       model_path="score_predictor_sequential.pth",
                       confidence_threshold=0.65, chunk_len=256,
                       use_zone_filter=True, use_scoreboard=False,
                       game="game_3", exp_id=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gt_left, gt_right = load_gt_events(event_json)
    print(f"Ground truth: {len(gt_left)} left, {len(gt_right)} right")

    dataset = SequentialScoreDataset(cache_path, event_json, csv_path, chunk_len=chunk_len)
    model = ScorePredictorSequential(feature_dim=dataset.feature_dim + 3).to(device)
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

        frames_pred, type_logits, hidden, score = model(
            feats, score_in, hidden=hidden, teacher_event_type=None)
        probs = torch.softmax(type_logits, dim=-1)[0]
        pred_cls = torch.argmax(type_logits, dim=-1)[0]
        frames_until = frames_pred[0, :, 0] * dataset.max_frames_ahead

        for i, real_frame in enumerate(frame_numbers):
            cls = pred_cls[i].item()
            conf = probs[i, cls].item()
            # this arm's speciality: shift the firing onto its predicted event frame
            fu = frames_until[i].item()
            frame_out = int(real_frame) + (max(0, int(fu)) if cls != 0 else 0)
            per_frame.append((frame_out, int(cls), float(conf)))

    if use_scoreboard:
        import cv2
        with open("data/OpenTT_Preprocess/video_bboxes.json") as f:
            bboxes = json.load(f)[game]
        split = "test" if game.startswith("test") else "train"
        cap = cv2.VideoCapture(f"data/OpenTT/videos/{split}/{game}.mp4")
        adjusted = []
        for frame, cls, conf in per_frame:
            if cls != 0 and conf >= confidence_threshold:
                ok = verify_visual_score_change(cap, frame, cls, bboxes, max_frames_ahead=240)
                conf = min(conf + 0.15, 1.0) if ok else max(conf - 0.40, 0.0)
            adjusted.append((frame, cls, conf))
        cap.release()
        per_frame = adjusted

    metrics = evaluate_stage2(per_frame, gt_left, gt_right, dataset.coords_dict,
                              min_confidence=confidence_threshold,
                              use_zone_filter_at_predict=use_zone_filter,
                              max_frames_ahead=dataset.max_frames_ahead, verbose=True)
    if exp_id:
        append_result(exp_id, "reg", None, metrics,
                      {"model_path": model_path, "min_conf": confidence_threshold,
                       "zone_predict": use_zone_filter, "scoreboard": use_scoreboard,
                       "csv_path": csv_path}, game=game)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="game_3")
    ap.add_argument("--model", default="score_predictor_sequential.pth")
    ap.add_argument("--csv-dir", default="runs/20260704-1040_baseline")
    ap.add_argument("--min-conf", type=float, default=0.65)
    ap.add_argument("--zone-filter", type=int, default=1)
    ap.add_argument("--scoreboard", action="store_true")
    ap.add_argument("--exp-id", default=None)
    a = ap.parse_args()
    split = "test" if a.game.startswith("test") else "train"
    predict_regression(
        f"data/dino_cache/{a.game}/cache.pt",
        f"data/OpenTT/annotations/{split}/{a.game}.json",
        f"{a.csv_dir}/{a.game}_predictions.csv",
        model_path=a.model, confidence_threshold=a.min_conf,
        use_zone_filter=bool(a.zone_filter), use_scoreboard=a.scoreboard,
        game=a.game, exp_id=a.exp_id)
