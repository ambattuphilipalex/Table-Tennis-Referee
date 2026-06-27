import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path


device = "mps"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


VIDEO_PATH = "/Users/philipalexambattu/Documents/DL Lab Final/Table Tennis Referee/datasets/OpenTT/videos/train/game_1.mp4"
BALL_JSON = "/Users/philipalexambattu/Documents/DL Lab Final/Table Tennis Referee/datasets/OpenTT/annotations/train/game_1_ball.json"

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def frame2tokens(frames):
    with torch.no_grad():
        x = frames.to(device)
        x = (x - MEAN.to(device)) / STD.to(device)
        out = model.forward_features(x)
        return out["x_norm_patchtokens"]

def read_frame(frame_no):
    cur = cv2.VideoCapture(VIDEO_PATH)
    frame_no = int(frame_no)
    cur.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    flag,frame = cur.read()
    cur.release()
    if flag == False:
        raise ValueError(f"couldn't read frame {frame_no} from {VIDEO_PATH}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224,224))
    frame = frame.astype("float32")
    frame = frame / 255
    frame = np.transpose(frame, axes=(2,0,1))
    frame = torch.tensor(frame)    
    return frame


class BallFrameDataset(Dataset):
    
    def __init__(self):
        cur = cv2.VideoCapture(VIDEO_PATH)
        height = cur.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cur.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = int(height)
        self.width = int(width)
        cur.release()
        
        with open(BALL_JSON) as f:
            ball_json = json.load(f)
        self.items = []
        for key,val in ball_json.items():
            if val['x'] == -1 and val['y'] == -1:
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
        frame = read_frame(frame_no)        
        ball_co_ordinates = torch.tensor([nx,ny],dtype=torch.float32)
        return frame, ball_co_ordinates, frame_no


if __name__ == "__main__":
    
    d = BallFrameDataset()
    # print(len(d))
    frame, ball, frame_no = d[0]
    # print(frame.shape, ball)
    



    model = torch.hub.load("facebookresearch/dinov2","dinov2_vits14")
    model = model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model = model.to(device)
    
    frame = read_frame(1000)
    batch = frame.unsqueeze(0)      
    tokens = frame2tokens(batch)
    print(tokens.shape)             
    
    cache_dir = Path("./dino_cache/game_1")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    frame_track = []
    
    loader = DataLoader(d,batch_size=32,shuffle=False)
    
    for frame, ball, frame_no in loader:
        tokens = frame2tokens(frame)
        tokens = tokens.cpu()
        for j in range (len(frame_no)): #within batch
            torch.save({"tokens": tokens[j], "ball": ball[j]}, cache_dir / f"{int(frame_no[j])}.pt")
            frame_track.append(int(frame_no[j]))
    
    with open( cache_dir/"index.json", "w") as  f:
        json.dump(frame_track,f)
        
    # verify: load one saved file back and check it round-trips
    one = torch.load(cache_dir / f"{frame_track[0]}.pt")
    print("tokens:", one["tokens"].shape)   
    print("ball:", one["ball"])             
        
        
        
        
    