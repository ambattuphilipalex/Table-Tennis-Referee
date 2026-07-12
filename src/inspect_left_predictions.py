import sys
import torch
from score_head import ScorePredictor
from score_dataset import SequenceScoreDataset
from score_eval_utils import load_gt_events

GAME_ID = sys.argv[1] if len(sys.argv) > 1 else "3"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inspecting game_{GAME_ID}")

    cache_path = f"data/dino_cache/game_{GAME_ID}/cache.pt"
    event_json = f"data/OpenTT/annotations/train/game_{GAME_ID}.json"
    csv_path = f"runs/20260704-1040_baseline/game_{GAME_ID}_predictions.csv"

    gt_left, gt_right = load_gt_events(event_json)
    dataset = SequenceScoreDataset(cache_path, event_json, csv_path)
    inferred_input_dim = dataset[0][0].shape[-1]

    model = ScorePredictor(input_dim=inferred_input_dim).to(device)
    model.load_state_dict(torch.load("score_predictor.pth", map_location=device))
    model.eval()

    frame_to_widx = {}
    for widx, start in enumerate(dataset.window_starts):
        last_idx = dataset.valid_indices[start + dataset.seq_len - 1]
        frame_to_widx[int(dataset.frames[last_idx])] = widx

    sorted_frames = sorted(frame_to_widx.keys())

    def nearest_window(target_frame):
        import bisect
        pos = bisect.bisect_left(sorted_frames, target_frame)
        candidates = []
        if pos < len(sorted_frames):
            candidates.append(sorted_frames[pos])
        if pos > 0:
            candidates.append(sorted_frames[pos - 1])
        if not candidates:
            return None
        best = min(candidates, key=lambda f: abs(f - target_frame))
        return best, frame_to_widx[best]

    print(f"{'real frame':>10} {'nearest window frame':>20} {'dist':>6} "
          f"{'P(bg)':>8} {'P(left)':>8} {'P(right)':>9}  argmax")
    with torch.no_grad():
        for gf in gt_left:
            result = nearest_window(gf)
            if result is None:
                print(f"{gf:>10}  no window covers this region")
                continue
            wframe, widx = result
            tokens, _ = dataset[widx]
            tokens = tokens.unsqueeze(0).to(device).float()
            logits = model(tokens)
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
            pred = int(torch.argmax(logits, dim=1).item())
            label = {0: "bg", 1: "LEFT", 2: "right"}[pred]
            print(f"{gf:>10} {wframe:>20} {abs(wframe-gf):>6} "
                  f"{probs[0]:>8.3f} {probs[1]:>8.3f} {probs[2]:>9.3f}  {label}")

    print("\nInterpretation:")
    print("- If P(left) is consistently close to P(right) (just losing the argmax),")
    print("  that's a calibration issue -- more training or class-weight tuning may fix it.")
    print("- If P(left) is consistently near 0 while P(bg)/P(right) dominate,")
    print("  the model isn't learning a Left signal at all -- check whether Left")
    print("  training examples are visually/positionally distinguishable from Right")
    print("  ones in your features, or whether there's a labeling issue.")


if __name__ == "__main__":
    main()