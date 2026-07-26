import torch
from torch.utils.data import DataLoader
import cv2
import sys
sys.path.append('src')

from score_head_regression import ScorePredictorSequential
from score_dataset_regression import SequentialScoreDataset
from score_eval_utils import load_gt_events

VIDEO_PATH = "/home/shared/Table-Tennis-Referee/data/OpenTT/videos/train/game_3.mp4"
OUTPUT_PATH = "annotated_output.mp4"
MODEL_PATH = "score_predictor_sequential.pth"
CACHE_PATH = "data/dino_cache/game_3/cache.pt"
EVENT_JSON = "data/OpenTT/annotations/train/game_3.json"
CSV_PATH = "runs/20260704-1040_baseline/game_3_predictions.csv"

CONFIDENCE_THRESHOLD = 0.70
HOLD_FRAMES = 60 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading dataset...")
dataset = SequentialScoreDataset(CACHE_PATH, EVENT_JSON, CSV_PATH, chunk_len=256)
feature_dim = dataset.feature_dim + 3 

print("Loading model...")
model = ScorePredictorSequential(feature_dim=feature_dim).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Running predictions...")
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
            if cls != 0 and conf >= CONFIDENCE_THRESHOLD and 0 <= fu <= dataset.max_frames_ahead:
                predicted_event_frame = real_frame + max(0, int(fu))
                raw_predictions.append({
                    "predicted_frame": predicted_event_frame,
                    "event_type": cls,
                    "confidence": conf,
                })

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

DETECTIONS = [
    (d["predicted_frame"], "LEFT" if d["event_type"] == 1 else "RIGHT", d["confidence"])
    for d in detections
]

print(f"Found {len(DETECTIONS)} events:")
for d in DETECTIONS:
    print(f"  {d[1]} at frame {d[0]} (conf={d[2]:.3f})")

print("\nCreating overlay video...")
detections_sorted = sorted(DETECTIONS, key=lambda d: d[0])

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Could not open video: {VIDEO_PATH}")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

left_score, right_score = 0, 0
frame_no = 0
next_det_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    while next_det_idx < len(detections_sorted) and detections_sorted[next_det_idx][0] <= frame_no:
        _, label, _ = detections_sorted[next_det_idx]
        if label == "LEFT":
            left_score += 1
        else:
            right_score += 1
        next_det_idx += 1

    cv2.putText(frame, f"LEFT {left_score} - {right_score} RIGHT", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    for det_frame, label, conf in detections_sorted:
        if det_frame <= frame_no < det_frame + HOLD_FRAMES:
            text = f"{label} POINT! ({conf:.2f})"
            color = (0, 255, 255) if label == "LEFT" else (255, 0, 255)
            cv2.putText(frame, text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    writer.write(frame)
    frame_no += 1
    if frame_no % 5000 == 0:
        print(f"  processed {frame_no}/{total_frames} frames...")

cap.release()
writer.release()
print(f"\nDone! Saved to {OUTPUT_PATH}")