import torch
from torch.utils.data import DataLoader
from score_head_regression import ScorePredictorSequential
from score_dataset_regression import SequentialScoreDataset
from score_eval_utils import load_gt_events

TEST_GAME = "4"  

def predict_regression(cache_path, event_json, csv_path,
                        model_path="score_predictor_sequential.pth",
                        confidence_threshold=0.70, chunk_len=256):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gt_left, gt_right = load_gt_events(event_json)
    true_left, true_right = len(gt_left), len(gt_right)
    print(f"Ground Truth: {true_left} Left, {true_right} Right")

    dataset = SequentialScoreDataset(cache_path, event_json, csv_path, chunk_len=chunk_len)
    feature_dim = dataset.feature_dim + 3

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
            probs = torch.softmax(type_logits, dim=-1)[0]
            pred_cls = torch.argmax(type_logits, dim=-1)[0]
            frames_until = frames_pred[0, :, 0] * dataset.max_frames_ahead

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

    print(f"Raw predictions (filtered): {len(raw_predictions)}")
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
    predict_regression(
        f"data/dino_cache/test_{TEST_GAME}/cache.pt",
        f"data/OpenTT/annotations/test/test_{TEST_GAME}.json",
        f"runs/20260704-1040_baseline/game_3_predictions.csv",
    )