import torch
from torch.utils.data import DataLoader

from score_head_regression import ScorePredictorSequential
from score_dataset_regression import SequentialScoreDataset
from score_eval_utils import load_gt_events

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
CLUSTER_WINDOWS = [60, 90, 120, 180]


@torch.no_grad()
def collect_raw_predictions(model, dataset, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    hidden, score = None, None
    rows = []

    for batch in loader:
        if bool(batch["is_run_start"].item()):
            hidden, score = None, None

        feats = batch["features"].to(device)
        frame_numbers = batch["frame_numbers"][0].tolist()
        score_in = score if score is not None else batch["score_state"].to(device)

        frames_pred, type_logits, hidden, score = model(
            feats, score_in, hidden=hidden, teacher_event_type=None
        )
        probs = torch.softmax(type_logits, dim=-1)[0]
        pred_cls = torch.argmax(type_logits, dim=-1)[0]
        frames_until = frames_pred[0, :, 0] * dataset.max_frames_ahead

        for i, real_frame in enumerate(frame_numbers):
            cls = pred_cls[i].item()
            conf = probs[i, cls].item()
            fu = frames_until[i].item()
            rows.append((real_frame, cls, conf, fu))

    return rows


def cluster_and_match(rows, threshold, cluster_window, cross_suppress_window,
                       gt_left, gt_right, max_frames_ahead=90):
    raw_predictions = []
    for real_frame, cls, conf, fu in rows:
        if cls != 0 and conf >= threshold and 0 <= fu <= max_frames_ahead:
            predicted_event_frame = real_frame + max(0, int(fu))
            raw_predictions.append({"predicted_frame": predicted_event_frame, "event_type": cls, "confidence": conf})

    raw_predictions.sort(key=lambda x: x["predicted_frame"])
    detections = []
    for cls in (1, 2):
        cls_preds = [p for p in raw_predictions if p["event_type"] == cls]
        if not cls_preds:
            continue
        cluster = [cls_preds[0]]
        for p in cls_preds[1:]:
            if p["predicted_frame"] - cluster[-1]["predicted_frame"] <= cluster_window:
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
        while j < len(detections) and detections[j]["predicted_frame"] - current["predicted_frame"] < cross_suppress_window:
            if detections[j]["event_type"] != current["event_type"] and detections[j]["confidence"] > current["confidence"]:
                current = detections[j]
            j += 1
        filtered.append(current)
        i = j
    detections = sorted(filtered, key=lambda x: x["predicted_frame"])

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

    total_gt = len(gt_left) + len(gt_right)
    recall = len(matched_gt) / total_gt if total_gt else 0.0
    precision = hits / (hits + misses) if (hits + misses) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return len(detections), hits, misses, recall, precision, f1


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_path = "data/dino_cache/game_3/cache.pt"
    event_json = "data/OpenTT/annotations/train/game_3.json"
    csv_path = "runs/20260704-1040_baseline/game_3_predictions.csv"
    model_path = "score_predictor_sequential.pth"

    gt_left, gt_right = load_gt_events(event_json)
    dataset = SequentialScoreDataset(cache_path, event_json, csv_path)
    feature_dim = dataset.feature_dim + 5

    model = ScorePredictorSequential(feature_dim=feature_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Running model forward ONCE to cache all per-frame predictions...")
    rows = collect_raw_predictions(model, dataset, device)
    print(f"Cached {len(rows)} per-frame predictions.\n")

    print(f"{'thresh':>7} {'clust_w':>8} {'det':>5} {'hits':>5} {'miss':>5} "
          f"{'recall':>8} {'precision':>10} {'f1':>7}")
    best = None
    for cw in CLUSTER_WINDOWS:
        for t in THRESHOLDS:
            n_det, hits, misses, recall, precision, f1 = cluster_and_match(
                rows, t, cw, cross_suppress_window=cw * 2 + 60,
                gt_left=gt_left, gt_right=gt_right,
            )
            print(f"{t:>7.2f} {cw:>8d} {n_det:>5d} {hits:>5d} {misses:>5d} "
                  f"{recall*100:>7.1f}% {precision*100:>9.1f}% {f1:>7.3f}")
            if best is None or f1 > best[0]:
                best = (f1, t, cw, recall, precision)

    f1, t, cw, recall, precision = best
    print(f"\nBest: threshold={t:.2f}, cluster_window={cw} -> "
          f"F1={f1:.3f} (recall={recall*100:.1f}%, precision={precision*100:.1f}%)")


if __name__ == "__main__":
    main()