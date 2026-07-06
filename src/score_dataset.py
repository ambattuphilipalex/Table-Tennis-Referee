import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class SequenceScoreDataset(Dataset):
    def __init__(self, ball_json_path, event_json_path, seq_len=64, dim=384):
        # Continuous timeline
        with open(ball_json_path, 'r') as f:
            ball_data = json.load(f)
        
        # Filter out frames where the ball wasn't found (-1, -1)
        self.valid_frames = sorted([
            int(k) for k, v in ball_data.items() 
            if v['x'] != -1 and v['y'] != -1
        ])
        
        with open(event_json_path, 'r') as f:
            self.events = json.load(f)
            
        self.seq_len = seq_len
        self.dim = dim
        
        self.left_scores = ["left_winner", "right_out", "right_net", "right_miss", "right_not_hitting"]
        self.right_scores = ["right_winner", "left_out", "left_net", "left_miss", "left_not_hitting"]

    def __len__(self):
        #at least 64 consecutive valid frames
        return max(0, len(self.valid_frames) - self.seq_len + 1)

    def __getitem__(self, idx):
        window_frames = self.valid_frames[idx : idx + self.seq_len]
        
        last_frame_str = str(window_frames[-1])
        event_str = self.events.get(last_frame_str, "empty_event")
        
        label = 0 # Default No score
        if any(score in event_str for score in self.left_scores):
            label = 1
        elif any(score in event_str for score in self.right_scores):
            label = 2
            
        fake_tokens = torch.randn(self.seq_len, self.dim)
        
        return fake_tokens, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    ball_json = "data/OpenTT/annotations/train/game_1_ball.json"
    event_json = "data/OpenTT/annotations/train/game_1.json"
    
    dataset = SequenceScoreDataset(ball_json, event_json)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for tokens, labels in loader:
        print(f"Tokens Batch Shape: {tokens.shape}")
        print(f"Labels Batch Shape: {labels.shape}")
        
        if labels.sum() > 0:
            print(f"Found scoring events in this batch! Labels: {labels}")
        break