import torch
import torch.nn as nn


class ScorePredictorClsSequential(nn.Module):
    def __init__(self, feature_dim=389, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        input_dim = feature_dim + 2 + 3  # + score_state(2) + prev-step event probs(3)
        self.cell = nn.GRUCell(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 3)  # 0=none, 1=left, 2=right

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, features, score_state, hidden=None):
        B, L, _ = features.shape
        device = features.device
        if hidden is None:
            hidden = self.init_hidden(B, device)

        prev_probs = torch.zeros(B, 3, device=device)
        prev_probs[:, 0] = 1.0 

        score = score_state.clone()
        type_out = []

        for t in range(L):
            x_t = torch.cat([features[:, t], score, prev_probs], dim=-1)
            hidden = self.cell(x_t, hidden)
            h = self.dropout(self.norm(hidden))

            type_t = self.fc(h)  # [B, 3]
            type_out.append(type_t)

            cls_t = torch.argmax(type_t, dim=-1)
            prev_probs = torch.softmax(type_t.detach(), dim=-1)

            left_inc = (cls_t == 1).float()
            right_inc = (cls_t == 2).float()
            score = torch.stack([score[:, 0] + left_inc, score[:, 1] + right_inc], dim=-1)

        type_out = torch.stack(type_out, dim=1)  # [B, L, 3]
        return type_out, hidden.detach(), score.detach()


if __name__ == "__main__":
    model = ScorePredictorClsSequential(feature_dim=389)
    fake_feats = torch.randn(2, 10, 389)
    fake_score = torch.tensor([[3.0, 5.0], [0.0, 0.0]])
    type_logits, hidden, score_out = model(fake_feats, fake_score)
    print("type_logits:", type_logits.shape)  # [2, 10, 3]
    print("hidden:", hidden.shape)              # [2, 128]
    print("score after chunk:", score_out)