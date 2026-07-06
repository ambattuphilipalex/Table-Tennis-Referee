import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from score_head import ScorePredictor
from score_dataset import SequenceScoreDataset

def train_exoskeleton():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    ball_json = "data/OpenTT/annotations/train/game_1_ball.json"
    event_json = "data/OpenTT/annotations/train/game_1.json"
    
    dataset = SequenceScoreDataset(ball_json, event_json)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = ScorePredictor().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # class Weights (The Imbalance Fix)
    # Class 0 (No Score) gets a tiny weight. Classes 1 & 2 get massive weights.
    class_weights = torch.tensor([0.01, 10.0, 10.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    epochs = 5
    print("Starting Exoskeleton Training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (tokens, labels) in enumerate(loader):
            tokens, labels = tokens.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(tokens)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Backward pass & optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train_exoskeleton()