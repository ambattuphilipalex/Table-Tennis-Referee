import torch
import torch.nn as nn


class ScorePredictorClsSequential(nn.Module):
    def __init__(self, feature_dim=389, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(feature_dim, hidden_dim,
                          bidirectional=True, batch_first=True)
        out_dim = hidden_dim * 2
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 3)  # 0=none, 1=left, 2=right

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, features, score_state, hidden=None):
        out, _ = self.gru(features)                    # [B, L, 2H]
        h = self.dropout(self.norm(out))
        type_out = self.fc(h)                          # [B, L, 3]
        return type_out, None, score_state.detach()


if __name__ == "__main__":
    model = ScorePredictorClsSequential(feature_dim=389)
    out = model(torch.randn(2, 10, 389), torch.tensor([[3.0, 5.0], [0.0, 0.0]]))
    print("type_logits:", tuple(out[0].shape))   # (2, 10, 3)
    print("hidden:", out[1])                      # None -- not carried across chunks
    print("score passthrough:", tuple(out[2].shape))