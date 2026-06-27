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

D_V_ROOT = "/home/shared/datasets/OpenTT"
D_A_ROOT = "/home/shared/Table-Tennis-Referee/data/OpenTT"
GAMES = ["game_1", "game_2", "game_3", "game_4", "game_5"]

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

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
    frame = cv2.resize(frame, (224,224))
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
        frame = read_frame(frame_no,self.video_path)        
        ball_co_ordinates = torch.tensor([nx,ny],dtype=torch.float32)
        return frame, ball_co_ordinates, frame_no


if __name__ == "__main__":

    model = torch.hub.load("facebookresearch/dinov2","dinov2_vits14")
    model = model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model = model.to(device)
         
    for game in GAMES:
        
        video_path = f"{D_V_ROOT}/videos/train/{game}.mp4"
        ball_json = f"{D_A_ROOT}//annotations/train/{game}_ball.json"
        d = BallFrameDataset(video_path,ball_json)

        cache_dir = Path(F"./dino_cache/{game}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        frame_track = []
        
        loader = DataLoader(d,batch_size=32,shuffle=False)
        
        for frame, ball, frame_no in loader:
            tokens = frame2tokens(frame)
            tokens = tokens.cpu()
            for j in range (len(frame_no)): #within batch
                torch.save({"tokens": tokens[j], "ball": ball[j]}, cache_dir / f"{int(frame_no[j])}.pt")
                frame_track.append(int(frame_no[j]))
        
        with open(cache_dir/"index.json", "w") as  f:
            json.dump(frame_track,f)
            
        # one = torch.load(cache_dir / f"{frame_track[0]}.pt")
        # print("tokens",one["tokens"].shape)   
        # print("ball",one["ball"])             
        
        
        
        
    
