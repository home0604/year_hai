# -*- coding: utf-8 -*-
"""
Flow Matching velocity-field backbones for 10x10 patch-antenna inverse design.

velocity field:  v_theta(x_t, t, cond)  ->  predicts u_t (target velocity)
  - x_t  : (B, num_bits)  noised pattern at time t (continuous, ~{-1,+1} target)
  - t    : (B,)           time in [0, 1]
  - cond : conditioning vector built from (spectral encoder output + time embedding)

1차(baseline) backbone:
  - 'mlp'  : flatten 100-dim -> residual MLP blocks with FiLM(adaLN-zero) conditioning
  - 'conv' : (1,10,10) -> conv blocks with per-channel FiLM conditioning

조건 주입(FiLM / adaLN-zero)이 품질을 좌우하므로 backbone은 갈아끼울 수 있게 분리.
나중에 DiT로 교체 시 build_backbone에 추가만 하면 됨.
"""
import math
import torch
import torch.nn as nn


# ---------------------------
# Sinusoidal timestep embedding
# ---------------------------
def timestep_embedding(t, dim, max_period=10000.0):
    """
    t: (B,) in [0,1] -> (B, dim) sinusoidal embedding.
    """
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    # t를 [0,1]에서 좀 더 분해능 좋게: *1000 스케일
    args = t.float().unsqueeze(-1) * 1000.0 * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)   # (B, 2*half)
    if dim % 2 == 1:  # zero pad if odd
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# ---------------------------
# 1D ResNet encoder for spectral condition (train_inverse_10x10.py 재사용)
# ---------------------------
class ResNet1DEncoder(nn.Module):
    """y: (B,1,P) -> 1D ResNet -> global average pooling -> (B, d_model)"""
    def __init__(self, d_model, num_blocks=3, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.input_proj = nn.Conv1d(1, d_model, kernel_size=kernel_size, padding=padding)
        self.input_bn = nn.BatchNorm1d(d_model)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(d_model),
                nn.ReLU(inplace=True),
                nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(d_model),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.act = nn.ReLU(inplace=True)

    def forward(self, y):
        x = self.act(self.input_bn(self.input_proj(y)))
        for block in self.blocks:
            x = self.act(block(x) + x)
        return x.mean(dim=-1)


class ConditionEncoder(nn.Module):
    """
    스펙트럼 y: (B, num_points) -> (B, d_model)
    spectral_cond: 'linear' | 'mlp' | 'resnet1d'
    """
    def __init__(self, num_points, d_model, spectral_cond="resnet1d"):
        super().__init__()
        self.spectral_cond = spectral_cond
        if spectral_cond == "linear":
            self.enc = nn.Linear(num_points, d_model)
        elif spectral_cond == "mlp":
            self.enc = nn.Sequential(
                nn.Linear(num_points, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
        elif spectral_cond == "resnet1d":
            self.enc = ResNet1DEncoder(d_model=d_model, num_blocks=3, kernel_size=3)
        else:
            raise ValueError(f"Unknown spectral_cond: {spectral_cond}")

    def forward(self, y):
        if self.spectral_cond == "resnet1d":
            return self.enc(y.unsqueeze(1))   # (B,1,P) -> (B,d_model)
        return self.enc(y)                    # (B,d_model)


# ---------------------------
# FiLM (adaLN-zero) MLP block
# ---------------------------
class FiLMMLPBlock(nn.Module):
    """
    h = x
    h_norm = LayerNorm(h)
    scale, shift = FiLM(cond)            # zero-init -> 초기엔 항등에 가깝게
    h_mod = h_norm * (1+scale) + shift
    out = x + gate * MLP(h_mod)
    """
    def __init__(self, hidden, cond_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        # adaLN-zero: scale/shift/gate 3개를 cond에서 생성, zero-init
        self.film = nn.Linear(cond_dim, 3 * hidden)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, cond):
        scale, shift, gate = self.film(cond).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        h = self.fc2(self.drop(self.act(self.fc1(self.act(h)))))
        return x + gate * h


class MLPVelocity(nn.Module):
    """flatten(num_bits) -> residual FiLM-MLP -> num_bits"""
    def __init__(self, num_bits, cond_dim, hidden=512, num_blocks=6, dropout=0.0):
        super().__init__()
        self.in_proj = nn.Linear(num_bits, hidden)
        self.blocks = nn.ModuleList([
            FiLMMLPBlock(hidden, cond_dim, dropout=dropout) for _ in range(num_blocks)
        ])
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, num_bits)
        # 출력 zero-init: 초기 velocity ~ 0 (안정적 학습 시작)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_flat, cond):
        h = self.in_proj(x_flat)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out_proj(self.out_norm(h))


# ---------------------------
# FiLM conv block (per-channel modulation)
# ---------------------------
class FiLMConvBlock(nn.Module):
    def __init__(self, ch, cond_dim, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.norm = nn.GroupNorm(8 if ch % 8 == 0 else 1, ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size, padding=p)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size, padding=p)
        self.act = nn.SiLU()
        self.film = nn.Linear(cond_dim, 3 * ch)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, cond):
        scale, shift, gate = self.film(cond).chunk(3, dim=-1)  # (B,ch) each
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        gate = gate[:, :, None, None]
        h = self.norm(x) * (1 + scale) + shift
        h = self.conv2(self.act(self.conv1(self.act(h))))
        return x + gate * h


class ConvVelocity(nn.Module):
    """(B,num_bits) -> (B,1,H,W) -> conv FiLM blocks -> (B,num_bits). 다운샘플 없음(10x10이라 불필요)."""
    def __init__(self, height, width, cond_dim, ch=128, num_blocks=6, kernel_size=3):
        super().__init__()
        self.height = height
        self.width = width
        self.in_conv = nn.Conv2d(1, ch, kernel_size, padding=kernel_size // 2)
        self.blocks = nn.ModuleList([
            FiLMConvBlock(ch, cond_dim, kernel_size) for _ in range(num_blocks)
        ])
        self.out_norm = nn.GroupNorm(8 if ch % 8 == 0 else 1, ch)
        self.out_conv = nn.Conv2d(ch, 1, kernel_size, padding=kernel_size // 2)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        self.act = nn.SiLU()

    def forward(self, x_flat, cond):
        B = x_flat.size(0)
        x = x_flat.view(B, 1, self.height, self.width)
        h = self.in_conv(x)
        for blk in self.blocks:
            h = blk(h, cond)
        out = self.out_conv(self.act(self.out_norm(h)))
        return out.view(B, -1)


# ---------------------------
# Full velocity field: cond encoder + time embed + backbone
# ---------------------------
class FlowVelocityNet(nn.Module):
    def __init__(
        self,
        num_points,
        height,
        width,
        d_model=192,
        spectral_cond="resnet1d",
        backbone="mlp",
        hidden=512,
        num_blocks=6,
        dropout=0.0,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.num_bits = height * width

        self.cond_encoder = ConditionEncoder(num_points, d_model, spectral_cond)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.d_model = d_model
        cond_dim = d_model  # cond = spectral_vec + time_emb

        if backbone == "mlp":
            self.net = MLPVelocity(self.num_bits, cond_dim, hidden=hidden,
                                   num_blocks=num_blocks, dropout=dropout)
        elif backbone == "conv":
            self.net = ConvVelocity(height, width, cond_dim, ch=hidden,
                                    num_blocks=num_blocks, kernel_size=3)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x_t, t, y):
        """
        x_t: (B, num_bits) continuous
        t:   (B,) in [0,1]
        y:   (B, num_points) spectrum
        return: v: (B, num_bits)
        """
        cond_vec = self.cond_encoder(y)                              # (B,d_model)
        t_emb = self.time_mlp(timestep_embedding(t, self.d_model))   # (B,d_model)
        cond = cond_vec + t_emb
        return self.net(x_t, cond)
