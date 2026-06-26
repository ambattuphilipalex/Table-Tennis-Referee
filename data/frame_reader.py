import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.utils.data import Dataset



VIDEO_PATH = "/Users/philipalexambattu/Documents/DL Lab Final/Table Tennis Referee/datasets/OpenTT/videos/train/game_1.mp4"
BALL_JSON = "/Users/philipalexambattu/Documents/DL Lab Final/Table Tennis Referee/datasets/OpenTT/annotations/train/game_1_ball.json"

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
    print(len(d))
    frame, ball, frame_no = d[0]
    print(frame.shape, ball)
    
    