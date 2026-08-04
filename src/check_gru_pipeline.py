import sys
import json
import torch
from pathlib import Path

from score_constants import LEFT_SCORES, RIGHT_SCORES

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
CACHE_ROOT = DATA_ROOT / "dino_cache"
EVENT_ROOT = DATA_ROOT / "OpenTT" / "annotations" / "train"

EVENT_MATCH_TOLERANCE = 30 


def check_game(game):
    cache_path = CACHE_ROOT / game / "cache.pt"
    event_path = EVENT_ROOT / f"{game}.json"

    if not cache_path.exists():
        print(f"No cache at {cache_path}")
        return
    if not event_path.exists():
        print(f"No event json at {event_path}")
        return

    blob = torch.load(cache_path, mmap=True, weights_only=False)
    frames = [int(f) for f in blob["frames"].tolist()]

    feat_path = CACHE_ROOT / game / "cls_token_features.pt"
    if feat_path.exists():
        feat_frames = set(int(f) for f in torch.load(feat_path, weights_only=False)["frames"].tolist())
        before = len(frames)
        frames = sorted(f for f in frames if f in feat_frames)
        print(f"  (intersected with cls_token_features.pt: {len(frames)}/{before} frames usable "
              f"-- this matches what score_dataset.py actually sees)")
    else:
        frames = sorted(frames)
        print(f"  (no cls_token_features.pt yet -- run extract_features.py first for accurate numbers)")

    with open(event_path) as f:
        events = json.load(f)

    gt_frames = []
    for fno, ev in events.items():
        if any(s in ev for s in LEFT_SCORES) or any(s in ev for s in RIGHT_SCORES):
            gt_frames.append(int(fno))
    gt_frames.sort()

    print(f"\n=== {game} ===")
    print(f"Cached frames: {len(frames)}  (range {frames[0]}..{frames[-1]})")

    gaps = [b - a for a, b in zip(frames, frames[1:])]
    if gaps:
        gaps_sorted = sorted(gaps)
        n = len(gaps_sorted)
        print(f"Gap between consecutive cached frames: "
              f"mean={sum(gaps)/n:.1f}  median={gaps_sorted[n//2]}  "
              f"p90={gaps_sorted[int(n*0.9)]}  max={gaps_sorted[-1]}")
        big_gaps = sum(1 for g in gaps if g > 10)
        print(f"Gaps > 10 real frames: {big_gaps}/{n} ({100*big_gaps/n:.1f}%) "
              f"-- these are places a 'sequential window' silently jumps in time")

    print(f"Real scoring events: {len(gt_frames)}")
    covered = 0
    for gf in gt_frames:
        # nearest cached frame
        nearest = min(frames, key=lambda f: abs(f - gf)) if frames else None
        dist = abs(nearest - gf) if nearest is not None else None
        if dist is not None and dist <= EVENT_MATCH_TOLERANCE:
            covered += 1
        else:
            print(f"  event at frame {gf}: nearest cached frame is {nearest} "
                  f"({dist} frames away) -- {'OK' if dist and dist<=EVENT_MATCH_TOLERANCE else 'LIKELY UNLEARNABLE'}")
    print(f"Events with a cached frame within {EVENT_MATCH_TOLERANCE} frames: "
          f"{covered}/{len(gt_frames)} ({100*covered/max(1,len(gt_frames)):.1f}%)")
    if covered < len(gt_frames):
        print("^ Events not covered above CANNOT be learned no matter how good the GRU is --"
              " the visual/coordinate evidence for them isn't in the cache.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0].startswith("--tolerance="):
        EVENT_MATCH_TOLERANCE = int(args[0].split("=", 1)[1])
        args = args[1:]
    games = args or ["game_1", "game_2", "game_3", "game_4"]
    print(f"Using event_match_tolerance={EVENT_MATCH_TOLERANCE}")
    for g in games:
        check_game(g)