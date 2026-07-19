import torch
import torch.nn as nn


class ScorePredictorRegression(nn.Module):
    def __init__(self, input_dim=387, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=gru_dropout, bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head_dropout = nn.Dropout(dropout)
        
        # Two outputs: frames_until_event (regression), event_type (classification)
        self.fc_frames = nn.Linear(hidden_dim, 1)      # Regression: how many frames?
        self.fc_type = nn.Linear(hidden_dim, 3)         # Classification: 0=no event, 1=Left, 2=Right

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        last = self.head_dropout(self.norm(last))
        
        frames_until = torch.sigmoid(self.fc_frames(last))
        event_logits = self.fc_type(last)
        
        return frames_until, event_logits


if __name__ == "__main__":
    model = ScorePredictorRegression()
    fake = torch.randn(16, 64, 387)
    frames, types = model(fake)
    print(f"Input shape: {fake.shape}")
    print(f"Frames until event: {frames.shape}")
    print(f"Event type logits: {types.shape}")
