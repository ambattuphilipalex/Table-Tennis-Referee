import torch
from torch.utils.data import DataLoader
import argparse
from score_head_regression import ScorePredictorSequential
from score_dataset_regression import SequentialScoreDataset
from score_eval_utils import load_gt_events, verify_visual_score_change
from pathlib import Path
import json
import cv2


def predict_regression(cache_path, event_json, csv_path,
                       model_path="score_predictor_sequential.pth",
                       confidence_threshold=0.70, chunk_len=256,
                       game=None, video_path=None,
                       use_scoreboard_verification=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    game = game or Path(cache_path).parent.name
    if video_path is None:
        split = "train" if game.startswith("game") else "test"
        video_path = f"data/OpenTT/videos/{split}/{game}.mp4"

    gt_left, gt_right = load_gt_events(event_json)
    true_left, true_right = len(gt_left), len(gt_right)
    print(f"Ground Truth: {true_left} Left, {true_right} Right")

    dataset = SequentialScoreDataset(cache_path, event_json, csv_path, chunk_len=chunk_len)
    feature_dim = dataset.feature_dim + 3  # + ball coords (x, y, fresh)

    model = ScorePredictorSequential(feature_dim=feature_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("\nRunning predictions (fully autoregressive, chunks in order)...")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    hidden, score = None, None

    raw_predictions = []
    with torch.no_grad():
        for batch in loader:
            if bool(batch["is_run_start"].item()):
                hidden, score = None, None

            feats = batch["features"].to(device)
            frame_numbers = batch["frame_numbers"][0].tolist()
            score_in = score if score is not None else batch["score_state"].to(device)

            frames_pred, type_logits, hidden, score = model(
                feats, score_in, hidden=hidden, teacher_event_type=None
            )
            probs = torch.softmax(type_logits, dim=-1)[0]              # [L, 3]
            pred_cls = torch.argmax(type_logits, dim=-1)[0]             # [L]
            frames_until = frames_pred[0, :, 0] * dataset.max_frames_ahead  # [L]

            for i, real_frame in enumerate(frame_numbers):
                cls = pred_cls[i].item()
                conf = probs[i, cls].item()
                fu = frames_until[i].item()
                if cls != 0 and conf >= confidence_threshold and 0 <= fu <= dataset.max_frames_ahead:
                    predicted_event_frame = real_frame + max(0, int(fu))
                    raw_predictions.append({
                        "predicted_frame": predicted_event_frame,
                        "event_type": cls,
                        "confidence": conf,
                    })

    print(f"Raw predictions (unfiltered): {len(raw_predictions)}")
    raw_predictions.sort(key=lambda x: x["predicted_frame"])

    detections = []
    for cls in (1, 2):
        cls_preds = [p for p in raw_predictions if p["event_type"] == cls]
        if not cls_preds:
            continue
        cluster = [cls_preds[0]]
        for p in cls_preds[1:]:
            if p["predicted_frame"] - cluster[-1]["predicted_frame"] <= 120:
                cluster.append(p)
            else:
                detections.append(max(cluster, key=lambda x: x["confidence"]))
                cluster = [p]
        if cluster:
            detections.append(max(cluster, key=lambda x: x["confidence"]))

    detections.sort(key=lambda x: x["predicted_frame"])
    filtered = []
    i = 0
    while i < len(detections):
        current = detections[i]
        j = i + 1
        while j < len(detections) and detections[j]["predicted_frame"] - current["predicted_frame"] < 300:
            if detections[j]["event_type"] != current["event_type"]:
                if detections[j]["confidence"] > current["confidence"]:
                    current = detections[j]
            j += 1
        filtered.append(current)
        i = j
    detections = sorted(filtered, key=lambda x: x["predicted_frame"])

    if use_scoreboard_verification:
        print(f"\nClustered events to verify: {len(detections)}")
        with open("data/OpenTT_Preprocess/video_bboxes.json", "r") as f:
            all_bboxes = json.load(f)
        game_bboxes = all_bboxes[game]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video for verification: {video_path}")

        verified_detections = []
        for idx, d in enumerate(detections):
            confirmed = verify_visual_score_change(
                cap,
                d["predicted_frame"],
                d["event_type"],
                game_bboxes,
                max_frames_ahead=240
            )

            if confirmed:
                d["confidence"] = min(d["confidence"] + 0.15, 1.0)
                print(f"  [{idx+1}/{len(detections)}] VERIFIED: Point detected at frame {d['predicted_frame']}")
            else:
                d["confidence"] = max(d["confidence"] - 0.40, 0.0)
                print(f"  [{idx+1}/{len(detections)}] FAILED: Graphic didn't change at {d['predicted_frame']} (penalizing)")

            if d["confidence"] >= confidence_threshold:
                verified_detections.append(d)

        cap.release()
        detections = verified_detections
    else:
        print(f"\nScoreboard verification OFF — keeping all {len(detections)} clustered detections")

    left_count = sum(1 for d in detections if d["event_type"] == 1)
    right_count = sum(1 for d in detections if d["event_type"] == 2)

    print(f"\n{'='*60}\nDETECTED EVENTS:\n{'='*60}")
    for d in detections:
        label = "LEFT " if d["event_type"] == 1 else "RIGHT"
        print(f"  {label} at frame ~{d['predicted_frame']} (conf={d['confidence']:.3f})")

    hits = misses = 0
    matched_gt = set()
    for d in detections:
        gt_list = gt_left if d["event_type"] == 1 else gt_right
        match = None
        for f in gt_list:
            if f not in matched_gt and -30 <= (f - d["predicted_frame"]) <= 150:
                match = f
                break
        if match is not None:
            matched_gt.add(match)
            hits += 1
        else:
            misses += 1

    total_gt = true_left + true_right
    recall = len(matched_gt) / total_gt if total_gt else 0.0
    precision = hits / (hits + misses) if (hits + misses) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\n{'='*30}\nFINAL RESULTS\n{'='*30}")
    print(f"True Score:     {true_left} - {true_right}")
    print(f"Predicted:      {left_count} - {right_count}")
    print(f"Hits: {hits}, Misses: {misses}")
    print(f"Recall: {len(matched_gt)}/{total_gt} ({100*recall:.1f}%)")
    print(f"Precision: {100*precision:.1f}%")
    print(f"F1: {f1:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Run the trained sequential scoring model on a game and report the scoreline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--game", default="game_3")
    p.add_argument("--cache-root", default="data/dino_cache")
    p.add_argument("--runs-dir", default="runs/20260704-1040_baseline")
    p.add_argument("--verify-scoreboard", action="store_true",
                   help="cross-check detections against the on-screen scoreboard "
                        "(needs a visible scoreboard; off by default)")
    args = p.parse_args()

    split = "train" if args.game.startswith("game") else "test"
    cache_path = f"{args.cache_root}/{args.game}/cache.pt"
    event_json = f"data/OpenTT/annotations/{split}/{args.game}.json"
    csv_path   = f"{args.runs_dir}/{args.game}_predictions.csv"

    predict_regression(
        cache_path, event_json, csv_path,
        game=args.game,
        use_scoreboard_verification=args.verify_scoreboard,
    )