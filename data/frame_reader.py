import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


VIDEO_PATH = "/Users/philipalexambattu/Documents/DL Lab Final/Table Tennis Referee/datasets/OpenTT/videos/train/game_1.mp4"


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
    frame = torch.from_numpy(frame)
    print(frame.shape, frame.min(), frame.max())
    return frame