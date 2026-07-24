import torch
from torch.utils.data import DataLoader

from score_head import ScorePredictorClsSequential
from score_dataset import SequentialClsDataset
from score_eval_utils import load_gt_events, check_scoreboard_change


@torch.no_grad()
def predict_and_evaluate(cache_path, event_json, csv_path,
                          model_path="score_predictor_cls_sequential.pth",
                          min_confidence=0.65, chunk_len=256, max_frames_ahead=30,
                          use_zone_filter=True, use_scoreboard_verification=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gt_left_frames, gt_right_frames = load_gt_events(event_json)
    true_left, true_right = len(gt_left_frames), len(gt_right_frames)
    print(f"Ground Truth Loaded: {true_left} Left Points, {true_right} Right Points in this video.")
    print(f"  -> Real Left Points at frames: {gt_left_frames}")
    print(f"  -> Real Right Points at frames: {gt_right_frames}")

    dataset = SequentialClsDataset(cache_path, event_json, csv_path,
                                    chunk_len=chunk_len, max_frames_ahead=max_frames_ahead)

    model = ScorePredictorClsSequential(feature_dim=dataset.feature_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("\nStarting Auto-Grader (fully autoregressive, chunks in order)...")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    hidden, score = None, None

    raw = []  # (frame, pred_class, confidence)
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
            if use_zone_filter:
                x, y, _ = dataset.coords_dict.get(real_frame, [0.5, 0.0, 0.0])[:3]
                if not (0.15 < x < 0.85 and 0.15 < y < 0.85):
                    continue

            cls = pred_cls[i].item()
            conf = probs[i, cls].item()
            if cls != 0 and conf >= min_confidence:
                raw.append((real_frame, cls, conf))

    print(f"Raw predictions (post zone-filter, pre-threshold-refinement): {len(raw)}")

    if use_scoreboard_verification:
        verified = []
        for real_frame, pred_class, confidence in raw:
            confirmed = check_scoreboard_change(dataset, real_frame, num_frames=150)
            if confirmed:
                verified.append((real_frame, pred_class, min(confidence + 0.15, 1.0)))
            else:
                verified.append((real_frame, pred_class, max(confidence - 0.15, 0.0)))
        raw = [(f, c, conf) for f, c, conf in verified if conf >= min_confidence]
        print(f"After scoreboard verification + re-threshold: {len(raw)}")

    detections = []
    for cls in (1, 2):
        cls_preds = sorted((r for r in raw if r[1] == cls), key=lambda r: r[0])
        if not cls_preds:
            continue
        cluster = [cls_preds[0]]
        for r in cls_preds[1:]:
            if r[0] - cluster[-1][0] <= 90:
                cluster.append(r)
            else:
                detections.append(max(cluster, key=lambda c: c[2]))
                cluster = [r]
        if cluster:
            detections.append(max(cluster, key=lambda c: c[2]))

    detections.sort(key=lambda d: d[0])
    filtered = []
    i = 0
    while i < len(detections):
        current = detections[i]
        j = i + 1
        while j < len(detections) and detections[j][0] - current[0] < 180:
            if detections[j][1] != current[1] and detections[j][2] > current[2]:
                current = detections[j]
            j += 1
        filtered.append(current)
        i = j
    detections = sorted(filtered, key=lambda d: d[0])

    left_points = sum(1 for d in detections if d[1] == 1)
    right_points = sum(1 for d in detections if d[1] == 2)

    hits = misses = 0
    matched_gt_frames = set()
    for real_frame, pred_class, _conf in detections:
        gt_list = gt_left_frames if pred_class == 1 else gt_right_frames
        label_name = "Left" if pred_class == 1 else "Right"
        match = None
        for f in gt_list:
            if f not in matched_gt_frames and -30 <= (f - real_frame) <= 150:
                match = f
                break
        if match is not None:
            matched_gt_frames.add(match)
            hits += 1
            print(f"HIT! {label_name} point correctly found near frame {real_frame} (Real: {match})")
        else:
            misses += 1
            print(f" MISS (False Alarm)! Model imagined a {label_name} point at {real_frame}")

    total_gt = true_left + true_right
    recall = len(matched_gt_frames) / total_gt if total_gt else 0.0
    precision = hits / (hits + misses) if (hits + misses) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("\n" + "=" * 30)
    print("FINAL EVALUATION REPORT")
    print("=" * 30)
    print(f"True Final Score:      {true_left} - {true_right}")
    print(f"Predicted Final Score: {left_points} - {right_points}")
    print(f"Final score matches:   {(true_left, true_right) == (left_points, right_points)}")
    print(f"True Positives (Hits): {hits}")
    print(f"False Positives:       {misses}")
    print(f"Ground-truth points found (recall): {len(matched_gt_frames)}/{total_gt} ({recall*100:.1f}%)")
    print(f"Model Precision: {precision*100:.1f}%")
    print(f"F1: {f1:.3f}")


if __name__ == "__main__":
    from score_train import EVAL_GAME_ID
    predict_and_evaluate(
        f"data/dino_cache/game_{EVAL_GAME_ID}/cache.pt",
        f"data/OpenTT/annotations/train/game_{EVAL_GAME_ID}.json",
        f"runs/20260704-1040_baseline/game_{EVAL_GAME_ID}_predictions.csv",
        min_confidence=0.65,
    )