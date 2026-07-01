import cv2
import torch
import numpy as np
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path


# device = "mps"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

D_AV_ROOT = DATA_ROOT / "OpenTT"

GAMES = [("train", f"game_{i}") for i in range(1, 6)] + \
        [("test",  f"test_{i}") for i in range(1, 8)]

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

RES= 518

def frame2tokens(frames):
    with torch.no_grad():
        x = frames.to(device)
        x = (x - MEAN.to(device)) / STD.to(device)
        out = model.forward_features(x)
        return out["x_norm_patchtokens"]

def read_frame(frame_no,video_path):
    cur = cv2.VideoCapture(video_path)
    frame_no = int(frame_no)
    cur.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    flag,frame = cur.read()
    cur.release()
    if flag == False:
        raise ValueError(f"couldn't read frame {frame_no} from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (RES,RES))
    frame = frame.astype("float32")
    frame = frame / 255
    frame = np.transpose(frame, axes=(2,0,1))
    frame = torch.tensor(frame)    
    return frame


class BallFrameDataset(Dataset):
    
    def __init__(self,video_path,ball_json):
        
        self.video_path = video_path
        cur = cv2.VideoCapture(video_path)
        height = cur.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cur.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = int(height)
        self.width = int(width)
        cur.release()
        
        with open(ball_json) as f:
            ball_json = json.load(f)
        self.items = []
        for key,val in ball_json.items():
            if val['x'] == -1 or val['y'] == -1:
                continue
            nx = val['x']/self.width
            ny = val['y']/self.height
            self.items.append((int(key), nx, ny))                
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self,i):
        selected = self.items[i]
        frame_no = selected[0]
        nx = selected[1]
        ny = selected[2]
        frame = read_frame(frame_no,self.video_path)        
        ball_co_ordinates = torch.tensor([nx,ny],dtype=torch.float32)
        return frame, ball_co_ordinates, frame_no


if __name__ == "__main__":

    model = torch.hub.load("facebookresearch/dinov2","dinov2_vits14")
    model = model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model = model.to(device)

    for split, game in GAMES:
    
        cache_dir = Path(f"{DATA_ROOT}/dino_cache/{game}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "cache.pt"
        if cache_path.exists():
            print(f"{game}: already cached, skipping")
            continue

        video_path = f"{D_AV_ROOT}/videos/{split}/{game}.mp4"
        ball_json  = f"{D_AV_ROOT}/annotations/{split}/{game}_ball.json"
        d = BallFrameDataset(video_path, ball_json)
        M = len(d)
        if M == 0:
            print(f"{game}: no valid frames, skipping")
            continue

        loader = DataLoader(d, batch_size=32, shuffle=False, num_workers=8)

        tokens = ball = frames = None      
        write = 0

        for frame, b, frame_no in loader:
            tok = frame2tokens(frame).to(torch.float16).cpu()
            bsz, N, D = tok.shape

            if tokens is None:          
                tokens = torch.from_file(str(cache_dir / "tokens.bin"), shared=True,
                                        size=M * N * D, dtype=torch.float16).view(M, N, D)
                ball   = torch.from_file(str(cache_dir / "ball.bin"), shared=True,
                                        size=M * 2, dtype=torch.float32).view(M, 2)
                frames = torch.from_file(str(cache_dir / "frames.bin"), shared=True,
                                        size=M, dtype=torch.int64)

            tokens[write:write + bsz] = tok
            ball[write:write + bsz]   = b.to(torch.float32)
            frames[write:write + bsz] = frame_no.to(torch.int64)
            write += bsz
            print(f"  {game}: {write}/{M} frames are done")

        meta = {"video": game, "split": split, "num_frames": M,
                "token_shape": [N, D], "resolution": RES,
                "orig_wh": [d.width, d.height]}
        
        torch.save({"tokens": tokens, "ball": ball, "frames": frames, "meta": meta}, cache_path)

        for tmp in ["tokens.bin", "ball.bin", "frames.bin"]:
            (cache_dir / tmp).unlink(missing_ok=True)

        print(f"\n{game}: cached {M} frames in {cache_path} tokens {tuple(tokens.shape)} {tokens.dtype}\n")