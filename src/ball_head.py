import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class BallHead(nn.Module):
    def __init__(self, in_dim=768, dim=384, depth=4, heads=6):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim) if in_dim != dim else nn.Identity()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls, std=1e-6)
        self.blocks = nn.Sequential(*[Block(dim, heads, mlp_ratio=4.0, qkv_bias=True)
                                      for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2)

    def forward(self, tokens):
        x = self.proj(tokens)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = self.blocks(torch.cat([cls, x], dim=1))
        return torch.sigmoid(self.head(self.norm(x[:, 0])))
