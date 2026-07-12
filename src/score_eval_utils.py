import json
import torch
from torch.utils.data import DataLoader

LEFT_SCORES = ["left_winner", "right_out", "right_net", "right_miss", "right_not_hitting"]
RIGHT_SCORES = ["right_winner", "left_out", "left_net", "left_miss", "left_not_hitting"]


def load_gt_events(event_json_path):
    with open(event_json_path, 'r') as f:
        events = json.load(f)
    gt_left, gt_right = [], []
    for frame_no, event_str in events.items():
        if any(s in event_str for s in LEFT_SCORES):
            gt_left.append(int(frame_no))
        elif any(s in event_str for s in RIGHT_SCORES):
            gt_right.append(int(frame_no))
    return sorted(gt_left), sorted(gt_right)


@torch.no_grad()
def evaluate_model(model, dataset, gt_left_frames, gt_right_frames, device,
                    tolerance=150, merge_window=120, verbose=True, min_confidence=0.0):
    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    raw = []  # (frame, class, confidence) for every above-threshold prediction
    for idx, (tokens, _) in enumerate(loader):
        start = dataset.window_starts[idx]
        last_valid_idx = dataset.valid_indices[start + dataset.seq_len - 1]
        real_frame = int(dataset.frames[last_valid_idx])

        xy = dataset.coords_dict.get(real_frame, [0.5, 0.0, 0.0])
        x, y = xy[0], xy[1]
        if not (0.2 < x < 1.8 and 0.0 < y < 1.2): 
            continue

        tokens = tokens.to(device).float()
        logits = model(tokens)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
        confidence = probs[0, pred_class].item()

        if pred_class != 0 and confidence >= min_confidence:
            raw.append((real_frame, pred_class, confidence))

    detections = []
    for cls in (1, 2):
        cls_preds = sorted((r for r in raw if r[1] == cls), key=lambda r: r[0])
        cluster = []
        for r in cls_preds:
            if cluster and r[0] - cluster[-1][0] > merge_window:
                detections.append(max(cluster, key=lambda c: c[2]))
                cluster = []
            cluster.append(r)
        if cluster:
            detections.append(max(cluster, key=lambda c: c[2]))
    detections.sort(key=lambda d: d[0])

    left_points = sum(1 for d in detections if d[1] == 1)
    right_points = sum(1 for d in detections if d[1] == 2)

    hits = misses = 0
    matched_gt_frames = set()
    for real_frame, pred_class, _conf in detections:
        gt_list = gt_left_frames if pred_class == 1 else gt_right_frames
        label_name = "Left" if pred_class == 1 else "Right"
        match = next((f for f in gt_list
                      if f not in matched_gt_frames and abs(f - real_frame) <= tolerance), None)
        if match is not None:
            matched_gt_frames.add(match)
            hits += 1
            if verbose:
                print(f"HIT! {label_name} point correctly found near frame {real_frame} (Real: {match})")
        else:
            misses += 1
            if verbose:
                print(f" MISS (False Alarm)! Model imagined a {label_name} point at {real_frame}")

    total_gt = len(gt_left_frames) + len(gt_right_frames)
    recall = (len(matched_gt_frames) / total_gt) if total_gt else 0.0
    precision = (hits / (hits + misses)) if (hits + misses) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    if was_training:
        model.train()

    return {
        "left_points": left_points, "right_points": right_points,
        "hits": hits, "misses": misses,
        "matched_gt": len(matched_gt_frames), "total_gt": total_gt,
        "recall": recall, "precision": precision, "f1": f1,
    }