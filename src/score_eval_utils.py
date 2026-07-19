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


def check_scoreboard_change(dataset, event_frame, num_frames=150):
    lost_ball_count = 0
    
    for offset in range(num_frames):
        frame_no = event_frame + offset
        coords = dataset.coords_dict.get(frame_no)
        
        if coords is None:
            continue
        
        fresh = coords[2]
        if fresh < 0.5:
            lost_ball_count += 1
    
    return lost_ball_count >= 30


@torch.no_grad()
def evaluate_model(model, dataset, gt_left_frames, gt_right_frames, device,
                    tolerance=150, merge_window=90, verbose=True, min_confidence=0.0):
    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    raw = []
    for idx, (tokens, _) in enumerate(loader):
        start = dataset.window_starts[idx]
        last_valid_idx = dataset.valid_indices[start + dataset.seq_len - 1]
        real_frame = int(dataset.frames[last_valid_idx])

        xy = dataset.coords_dict.get(real_frame, [0.5, 0.0, 0.0])
        x, y = xy[0], xy[1]
        
        if not (0.15 < x < 0.85 and 0.15 < y < 0.85):
            continue

        tokens = tokens.to(device).float()
        logits = model(tokens)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
        confidence = probs[0, pred_class].item()

        if pred_class != 0 and confidence >= min_confidence:
            raw.append((real_frame, pred_class, confidence))

    verified_raw = []
    for real_frame, pred_class, confidence in raw:
        scoreboard_confirmed = check_scoreboard_change(dataset, real_frame, num_frames=150)
        
        if scoreboard_confirmed:
            verified_raw.append((real_frame, pred_class, min(confidence + 0.15, 1.0)))
            if verbose:
                print(f"  [Scoreboard Verified] Frame {real_frame}: confidence boosted")
        else:
            verified_raw.append((real_frame, pred_class, max(confidence - 0.15, 0.0)))
    
    raw = verified_raw

    detections = []
    for cls in (1, 2):
        cls_preds = sorted((r for r in raw if r[1] == cls), key=lambda r: r[0])
        if not cls_preds:
            continue
            
        cluster = [cls_preds[0]]
        for r in cls_preds[1:]:
            if r[0] - cluster[-1][0] <= merge_window:
                cluster.append(r)
            else:
                best = max(cluster, key=lambda c: c[2])
                detections.append(best)
                cluster = [r]
        if cluster:
            best = max(cluster, key=lambda c: c[2])
            detections.append(best)
    
    detections.sort(key=lambda d: d[0])
    
    filtered_detections = []
    i = 0
    while i < len(detections):
        current = detections[i]
        j = i + 1
        while j < len(detections) and detections[j][0] - current[0] < 180:
            if detections[j][1] != current[1]:
                if detections[j][2] > current[2]:
                    current = detections[j]
            j += 1
        filtered_detections.append(current)
        i = j
    
    detections = sorted(filtered_detections, key=lambda d: d[0])

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