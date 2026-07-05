import json
from pathlib import Path

import cv2
import torch

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
D_AV_ROOT = DATA_ROOT / "OpenTT"
CACHE_ROOT = DATA_ROOT / "dino_cache"
REVIEW_ROOT = DATA_ROOT / "bad_label_review"          
CAND_ROOT = DATA_ROOT / "bad_label_candidates"        

GAMES = [("train", f"game_{i}") for i in range(1, 6)] + \
        [("test",  f"test_{i}") for i in range(1, 8)]

THRESH_PX = 150.0


def read_frame_bgr(video_path, frame_no):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def flag_game(split, game):
    cache_path = CACHE_ROOT / game / "cache.pt"
    video_path = D_AV_ROOT / "videos" / split / f"{game}.mp4"
    if not cache_path.exists():
        print(f"{game}: no cache, skip"); return []
    if not video_path.exists():
        print(f"{game}: no video, skip"); return []

    blob = torch.load(cache_path, mmap=True, weights_only=False)
    ball = blob["ball"].float()                         
    frames = blob["frames"].long()                      
    wh = torch.tensor(blob.get("meta", {}).get("orig_wh", (1920, 1080)), dtype=torch.float32)
    px = ball * wh                                      


    flagged = []
    for i in range(1, len(px) - 1):
        gap_prev = int(frames[i] - frames[i - 1])
        gap_next = int(frames[i + 1] - frames[i])
        if gap_prev != 1 or gap_next != 1:              
            continue
        d_prev = (px[i] - px[i - 1]).norm().item()
        d_next = (px[i + 1] - px[i]).norm().item()
        if d_prev > THRESH_PX and d_next > THRESH_PX:   
            flagged.append((int(frames[i]), i, d_prev, d_next))

    out_dir = REVIEW_ROOT / game
    out_dir.mkdir(parents=True, exist_ok=True)
    for fno, i, d_prev, d_next in flagged:
        img = read_frame_bgr(video_path, fno)
        if img is None:
            continue
        cur = tuple(int(v) for v in px[i].tolist())         
        prv = tuple(int(v) for v in px[i - 1].tolist())     
        nxt = tuple(int(v) for v in px[i + 1].tolist())     
        cv2.circle(img, prv, 8, (0, 255, 0), 2)             
        cv2.circle(img, nxt, 8, (255, 0, 0), 2)             
        cv2.circle(img, cur, 12, (0, 0, 255), 3)            
        cv2.putText(img, f"f={fno}  jump_prev={d_prev:.0f}px  jump_next={d_next:.0f}px",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
        cv2.imwrite(str(out_dir / f"f{fno:07d}_jump{int(max(d_prev, d_next))}.png"), img)


    CAND_ROOT.mkdir(parents=True, exist_ok=True)
    cand = sorted(f for f, *_ in flagged)
    json.dump(cand, open(CAND_ROOT / f"{game}.json", "w"))
    print(f"{game}: flagged {len(cand)} / {len(px)} frames "
          f"({100*len(cand)/len(px):.2f}%)")
    return cand


if __name__ == "__main__":
    total = 0
    for split, game in GAMES:
        total += len(flag_game(split, game))
    print(f"{total} candidate frames flagged.")

