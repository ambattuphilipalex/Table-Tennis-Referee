import torch
from score_head_regression import ScorePredictorRegression
from score_dataset_regression import SequenceScoreDatasetRegression
from score_eval_utils import load_gt_events


def predict_regression(cache_path, event_json, csv_path, model_path="score_predictor_regression.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gt_left, gt_right = load_gt_events(event_json)
    true_left, true_right = len(gt_left), len(gt_right)
    print(f"Ground Truth: {true_left} Left, {true_right} Right")

    dataset = SequenceScoreDatasetRegression(cache_path, event_json, csv_path)
    sample_features, _ = dataset[0]
    input_dim = sample_features.shape[-1]

    model = ScorePredictorRegression(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("\nRunning predictions...")
    
    raw_predictions = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            tokens, _ = dataset[idx]
            tokens = tokens.unsqueeze(0).to(device).float()
            
            frames_pred, type_logits = model(tokens)
            frames_until = frames_pred.item() * dataset.max_frames_ahead
            event_probs = torch.softmax(type_logits, dim=1)[0]
            pred_type = torch.argmax(type_logits, dim=1).item()
            confidence = event_probs[pred_type].item()
            
            if pred_type != 0 and confidence >= 0.70 and 0 <= frames_until <= 90:
                start = dataset.window_starts[idx]
                last_idx = dataset.valid_indices[start + dataset.seq_len - 1]
                real_frame = int(dataset.frames[last_idx])
                predicted_event_frame = real_frame + max(0, int(frames_until))
                
                raw_predictions.append({
                    'predicted_frame': predicted_event_frame,
                    'event_type': pred_type,
                    'confidence': confidence
                })

    print(f"Raw predictions (filtered): {len(raw_predictions)}")

    raw_predictions.sort(key=lambda x: x['predicted_frame'])
    
    detections = []
    for cls in (1, 2):
        cls_preds = [p for p in raw_predictions if p['event_type'] == cls]
        if not cls_preds:
            continue
        
        cluster = [cls_preds[0]]
        for p in cls_preds[1:]:
            if p['predicted_frame'] - cluster[-1]['predicted_frame'] <= 120:
                cluster.append(p)
            else:
                best = max(cluster, key=lambda x: x['confidence'])
                detections.append(best)
                cluster = [p]
        if cluster:
            best = max(cluster, key=lambda x: x['confidence'])
            detections.append(best)
    
    detections.sort(key=lambda x: x['predicted_frame'])
    filtered = []
    i = 0
    while i < len(detections):
        current = detections[i]
        j = i + 1
        while j < len(detections) and detections[j]['predicted_frame'] - current['predicted_frame'] < 300:
            if detections[j]['event_type'] != current['event_type']:
                if detections[j]['confidence'] > current['confidence']:
                    current = detections[j]
            j += 1
        filtered.append(current)
        i = j
    
    detections = sorted(filtered, key=lambda x: x['predicted_frame'])

    left_count = sum(1 for d in detections if d['event_type'] == 1)
    right_count = sum(1 for d in detections if d['event_type'] == 2)
    
    print(f"\n{'='*60}")
    print(f"DETECTED EVENTS:")
    print(f"{'='*60}")
    for d in detections:
        label = "LEFT " if d['event_type'] == 1 else "RIGHT"
        print(f"  {label} at frame ~{d['predicted_frame']} (conf={d['confidence']:.3f})")

    # Match with ground truth
    hits = 0
    misses = 0
    matched_gt = set()
    
    for d in detections:
        gt_list = gt_left if d['event_type'] == 1 else gt_right
        match = None
        for f in gt_list:
            if f not in matched_gt and -30 <= (f - d['predicted_frame']) <= 150:
                match = f
                break
        if match:
            matched_gt.add(match)
            hits += 1
        else:
            misses += 1

    total_gt = true_left + true_right
    recall = len(matched_gt) / total_gt if total_gt > 0 else 0
    precision = hits / (hits + misses) if (hits + misses) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*30}")
    print(f"FINAL RESULTS")
    print(f"{'='*30}")
    print(f"True Score:     {true_left} - {true_right}")
    print(f"Predicted:      {left_count} - {right_count}")
    print(f"Hits: {hits}, Misses: {misses}")
    print(f"Recall: {len(matched_gt)}/{total_gt} ({100*recall:.1f}%)")
    print(f"Precision: {100*precision:.1f}%")
    print(f"F1: {f1:.3f}")


if __name__ == "__main__":
    # Test on game_3
    predict_regression(
        "data/dino_cache/game_3/cache.pt",
        "data/OpenTT/annotations/train/game_3.json",
        "runs/20260704-1040_baseline/game_3_predictions.csv",
    )