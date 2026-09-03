import numpy as np
import torch
import torch.nn as nn

from src.config import LOOKBACK
from torch_geometric.nn import GATv2Conv


def sinusoidal_pe(T: int, d: int) -> torch.Tensor:
    assert d % 2 == 0, "sinusoidal PE needs even dimension"
    pos = np.arange(T, dtype=float)[:, None]
    i = np.arange(d // 2, dtype=float)[None, :]
    ang = pos / np.power(10000.0, 2 * i / d)
    pe = np.zeros((T, d), dtype=np.float32)
    pe[:, 0::2] = np.sin(ang)
    pe[:, 1::2] = np.cos(ang)
    return torch.from_numpy(pe)


class THGNN(nn.Module):
    def __init__(self, d_feat, T=LOOKBACK, d_in=64, d_enc=64, d_att=64,
                 n_enc_heads=4, n_gat_heads=4, d_ff=128, d_hga=32,
                 dropout=0.3, relations=("pos", "neg", "sent")):
        super().__init__()
        self.T, self.relations = T, tuple(relations)
        self.in_proj = nn.Linear(d_feat, d_in)
        self.register_buffer("pe", sinusoidal_pe(T, d_in))
        enc_layer = nn.TransformerEncoderLayer(d_in, n_enc_heads, d_ff,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        flat = T * d_enc
        self.self_proj = nn.Linear(flat, d_att)
        self.rel_conv = nn.ModuleDict({
            r: GATv2Conv(flat, d_att, heads=n_gat_heads, concat=False, edge_dim=1)
            for r in self.relations})
        self.hga = nn.Linear(d_att, d_hga)
        self.q = nn.Parameter(torch.randn(d_hga) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(d_att, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x, edges, return_beta=False):
        B, n, T, _ = x.shape
        assert T == self.T
        inp = (self.in_proj(x) + self.pe[None, None]).reshape(B * n, T, -1)
        h = self.encoder(inp)
        h = self.drop(h).reshape(B * n, T * h.shape[-1])
        msgs = [self.self_proj(h)]
        for r in self.relations:
            ei, ew = edges[r]
            msgs.append(self.rel_conv[r](h, ei, edge_attr=ew.unsqueeze(-1)))
        M = torch.stack(msgs, dim=1)
        beta = torch.softmax(torch.tanh(self.hga(M)) @ self.q,
                             dim=1).unsqueeze(-1)
        Z = (beta * M).sum(dim=1)
        scores = self.head(self.drop(Z)).squeeze(-1).view(B, n)
        if return_beta:
            return scores, Z.view(B, n, -1), beta.view(B, n, -1)
        return scores, Z.view(B, n, -1)
