import torch
import torch.nn as nn


class ScorePredictor(nn.Module):
    def __init__(self, input_dim=387, hidden_dim=128, num_layers=2, num_classes=3,
                 bidirectional=True, dropout=0.3):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=gru_dropout, bidirectional=bidirectional,
        )
        out_dim = hidden_dim * 2
        self.norm = nn.LayerNorm(out_dim)
        self.head_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        logits = self.fc(self.head_dropout(self.norm(last)))
        return logits

if __name__ == "__main__":
    model = ScorePredictor()
    fake_fused_tokens = torch.randn(16, 64, 387)
    predictions = model(fake_fused_tokens)
    print(f"Input shape: {fake_fused_tokens.shape}")
    print(f"Fused output shape: {predictions.shape}")