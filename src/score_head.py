import torch
import torch.nn as nn

class ScorePredictor(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, num_layers=2, num_classes=3):
        super().__init__()
        # Batch, Sequence, Features
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        
        # Maps the final hidden state to our classes:
        # 0 = No score, 1 = Player A scores, 2 = Player B scores
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (B, T, D)
        out, hidden = self.gru(x)
        
        # out[:, -1, :] grabs the last frame's output for the whole batch
        logits = self.fc(out[:, -1, :]) 
        
        return logits

if __name__ == "__main__":
    model = ScorePredictor()
    
    batch_size = 16
    seq_len = 64
    dim = 384
    fake_cls_tokens = torch.randn(batch_size, seq_len, dim)
    
    predictions = model(fake_cls_tokens)
    print(f"Input shape: {fake_cls_tokens.shape}")
    print(f"Output shape: {predictions.shape}")