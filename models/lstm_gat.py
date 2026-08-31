"""
LSTM-GAT neural network architecture with multi-head spatial attention and decile allocation head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import config


class DenseGATLayer(nn.Module):
    """
    Dense Graph Attention Layer operating on batched (B, N, F_in) nodes
    and (B, N, N) binary adjacency matrices, with single-head or multi-head attention.
    """
    def __init__(
        self,
        in_features: int = 80,
        out_features: int = 80,
        heads: int = 1,
        dropout: float = 0.20,
        alpha: float = 0.15,
        activation: str = "elu"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.dropout = dropout
        self.alpha = alpha
        self.activation = activation

        assert out_features % heads == 0, f"out_features ({out_features}) must be divisible by heads ({heads})"
        self.head_dim = out_features // heads

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.head_dim, 1))
        self.a_dst = nn.Parameter(torch.empty(heads, self.head_dim, 1))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)
        self.dropout_layer = nn.Dropout(p=dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.xavier_uniform_(self.a_src, gain=gain)
        nn.init.xavier_uniform_(self.a_dst, gain=gain)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, return_attention: bool = False):
        """
        Args:
            h: Node features of shape (B, N, in_features)
            adj: Adjacency matrices of shape (B, N, N)
            return_attention: If True, returns (output, attention_weights)
        Returns:
            Output node features of shape (B, N, out_features)
        """
        B, N, _ = h.shape
        Wh = self.W(h)

        if self.heads == 1:
            f_src = torch.matmul(Wh, self.a_src[0])  # (B, N, 1)
            f_dst = torch.matmul(Wh, self.a_dst[0])  # (B, N, 1)

            e = self.leaky_relu(f_src + f_dst.transpose(1, 2))

            mask = (adj > 0.5)
            e = e.masked_fill(~mask, -1e9)

            alpha_attn = F.softmax(e, dim=-1)
            alpha_drop = self.dropout_layer(alpha_attn)

            h_prime = torch.bmm(alpha_drop, Wh) + self.bias
        else:
            Wh_heads = Wh.view(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)
            f_src = torch.matmul(Wh_heads, self.a_src.unsqueeze(0))
            f_dst = torch.matmul(Wh_heads, self.a_dst.unsqueeze(0))
            e = self.leaky_relu(f_src + f_dst.transpose(-2, -1))
            mask = (adj > 0.5).unsqueeze(1)
            e = e.masked_fill(~mask, -1e9)

            alpha_attn = F.softmax(e, dim=-1)
            alpha_drop = self.dropout_layer(alpha_attn)
            out_heads = torch.matmul(alpha_drop, Wh_heads)
            h_prime = out_heads.permute(0, 2, 1, 3).contiguous().view(B, N, self.out_features) + self.bias
            alpha_attn = alpha_attn.mean(dim=1)

        if self.activation == "elu":
            out = F.elu(h_prime)
        elif self.activation == "leaky_relu":
            out = self.leaky_relu(h_prime)
        else:
            out = h_prime

        if return_attention:
            return out, alpha_attn
        return out


class LSTMGATModel(nn.Module):
    """
    Model v4: LSTM-GAT for Portfolio Optimization.
    Produces dynamic allocation weights around equal-weight baseline,
    with support for full linear tilting or Top-K high-conviction concentration.
    """
    def __init__(
        self,
        num_features: int = config.NUM_FEATURES,       # 10
        lookback_r: int = config.LOOKBACK_R,           # 30
        num_assets: int = config.NUM_ASSETS,           # Dynamic (28)
        lstm_hidden: int = config.LSTM_HIDDEN_SIZE,    # 80
        lstm_dropout: float = config.LSTM_DROPOUT,     # 0.27
        gat_hidden: int = config.GAT_HIDDEN_SIZE,      # 80
        gat_layers: int = config.GAT_LAYERS,           # 2
        gat_heads: int = config.GAT_HEADS,             # 1
        gat_dropout: float = config.GAT_DROPOUT,       # 0.20
        leaky_relu_alpha: float = config.LEAKY_RELU_ALPHA,  # 0.15
        final_dropout: float = config.FINAL_DROPOUT,   # 0.29
        weight_norm_eps: float = config.WEIGHT_NORM_EPS, # 1e-8
        tilt_scale: Optional[float] = None,
        top_k: Optional[int] = None
    ):
        super().__init__()
        self.num_features = num_features
        self.lookback_r = lookback_r
        self.num_assets = num_assets
        self.gat_heads = gat_heads
        self.top_k = top_k
        # Scale-invariant tilt magnitude
        if tilt_scale is None:
            self.tilt_scale = 0.55 * math.sqrt(9.0 / max(num_assets, 1))
        else:
            self.tilt_scale = tilt_scale
        self.weight_norm_eps = weight_norm_eps

        # 1. Shared LSTM
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden,
            num_layers=config.LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=config.LSTM_BIDIRECTIONAL
        )
        self.lstm_dropout = nn.Dropout(p=lstm_dropout)

        # 2. Dynamic 2-Layer GAT with Residual Connections
        self.gat1 = DenseGATLayer(
            in_features=lstm_hidden,
            out_features=gat_hidden,
            heads=gat_heads,
            dropout=gat_dropout,
            alpha=leaky_relu_alpha,
            activation="elu"
        )
        self.gat2 = DenseGATLayer(
            in_features=gat_hidden,
            out_features=gat_hidden,
            heads=gat_heads,
            dropout=gat_dropout,
            alpha=leaky_relu_alpha,
            activation="elu"
        )

        # 3. Final Linear Layer + Tanh
        self.final_dropout = nn.Dropout(p=final_dropout)
        self.linear_out = nn.Linear(gat_hidden, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, return_intermediates: bool = False):
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, N, 30, 10)
            adj: Dynamic adjacency matrix of shape (B, N, N)
            return_intermediates: If True, returns dict with intermediate tensors.
        Returns:
            Predicted portfolio weights of shape (B, N) summing to 1.0 (or dict if requested).
        """
        B, N, R, F_in = x.shape

        # 1. Shared LSTM over all assets
        x_flat = x.view(B * N, R, F_in)
        _, (h_n, _) = self.lstm(x_flat)
        h_lstm = self.lstm_dropout(h_n.squeeze(0))
        h_nodes = h_lstm.view(B, N, -1)

        # 2. Dynamic 2-Layer GAT with Residual Skip-Connections
        g1_out, alpha1 = self.gat1(h_nodes, adj, return_attention=True)
        z1 = g1_out + h_nodes
        g2_out, alpha2 = self.gat2(z1, adj, return_attention=True)
        z2 = g2_out + z1

        # 3. Final Linear Layer + Tilting
        z_drop = self.final_dropout(z2)
        raw_out = self.linear_out(z_drop).squeeze(-1)  # (B, N)

        if self.top_k is not None and 0 < self.top_k <= N // 2:
            _, idx_top = torch.topk(raw_out, k=self.top_k, dim=-1, largest=True)
            _, idx_bot = torch.topk(raw_out, k=self.top_k, dim=-1, largest=False)

            top_w = F.softmax(torch.gather(raw_out, -1, idx_top), dim=-1)
            bot_w = F.softmax(-torch.gather(raw_out, -1, idx_bot), dim=-1)

            active_tilt = torch.zeros_like(raw_out)
            active_tilt.scatter_add_(-1, idx_top, top_w)
            active_tilt.scatter_add_(-1, idx_bot, -bot_w)

            tilt_final = active_tilt - active_tilt.mean(dim=-1, keepdim=True)
            weights = (1.0 / N) + self.tilt_scale * tilt_final
        else:
            tilt_raw = torch.tanh(raw_out)
            tilt_centered = tilt_raw - tilt_raw.mean(dim=-1, keepdim=True)
            weights = (1.0 / N) + self.tilt_scale * tilt_centered

        if return_intermediates:
            return {
                "weights": weights,
                "h_nodes": h_nodes,
                "z1": z1,
                "z2": z2,
                "alpha1": alpha1,
                "alpha2": alpha2
            }
        return weights


if __name__ == "__main__":
    config.set_seed(42)
    model = LSTMGATModel(num_assets=30)
    x_dummy = torch.randn(4, 30, 30, 10)
    adj_dummy = torch.eye(30).unsqueeze(0).repeat(4, 1, 1)
    w_out = model(x_dummy, adj_dummy)
    print(f"Output weights shape: {w_out.shape}")
    print(f"Weights sum per sample: {w_out.sum(dim=-1)}")
    print(f"Sample weights:\n{w_out[0].detach().numpy()}")
