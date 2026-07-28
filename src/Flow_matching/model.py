# -*- coding: utf-8 -*-
"""
Flow Matching DiT variants for 10x10 patch-antenna inverse design.

Two architectures:
  - Stable3DiT: SD3/MMDiT-style (adaLN-Zero + cross-attention + bidirectional Transformer)
  - SmallDiT:   AR-comparable (additive conditioning + standard TransformerEncoder, no adaLN)

Both share the same forward(x_t, t, y) / forward_uncond(x_t, t) interface.
"""
import math
import torch
import torch.nn as nn

from backbones import ResNet1DEncoder


# ---------------------------
# Sinusoidal timestep embedding
# ---------------------------
def timestep_embedding(t, dim, max_period=10000.0):
    """t: (B,) in [0,1] -> (B, dim) sinusoidal embedding."""
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(-1) * 1000.0 * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# ===========================================================================
# ResNet1D Sequence Encoder (for Stable3DiT cross-attention)
# ===========================================================================
class ResNet1DSeqEncoder(nn.Module):
    """y: (B,1,P) → 1D ResNet → AdaptiveAvgPool1d(seq_len_y) → (B, seq_len_y, d_model)"""
    def __init__(self, d_model, num_blocks=3, kernel_size=3, seq_len_y=50):
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
        self.pool = nn.AdaptiveAvgPool1d(seq_len_y)

    def forward(self, y):
        x = self.act(self.input_bn(self.input_proj(y)))
        for block in self.blocks:
            x = self.act(block(x) + x)
        x = self.pool(x)
        return x.transpose(1, 2)  # (B, seq_len_y, d_model)


# ===========================================================================
# Stable3DiT: SD3/MMDiT-style (adaLN-Zero + cross-attention)
# ===========================================================================
class Stable3DiTBlock(nn.Module):
    """
    adaLN-Zero modulated block. cond = adaLN 구동 벡터 (t, 또는 t+pooled-y).

    use_cross_attn=True (crossattn 모드):
        SA → Cross-Attn(y_seq) → FFN, 변조 7개 (γ₁β₁α₁, α₂, γ₃β₃α₃)
    use_cross_attn=False (adaln 모드, 정통 DiT식):
        SA → FFN, cross-attn 없음, 변조 6개 (γ₁β₁α₁, γ₃β₃α₃)
        → y 는 cond(t+pooled-y)를 통해 adaLN 으로만 주입
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, use_cross_attn=True):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        n_mod = 7 if use_cross_attn else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, n_mod * d_model),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        self.norm_sa = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        if use_cross_attn:
            self.norm_ca_q = nn.LayerNorm(d_model)
            self.norm_ca_kv = nn.LayerNorm(d_model)
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, cond, y_seq=None):
        mod = self.adaLN_modulation(cond)
        if self.use_cross_attn:
            gamma1, beta1, alpha1, alpha2, gamma3, beta3, alpha3 = mod.chunk(7, dim=-1)
            alpha2 = alpha2.unsqueeze(1)
        else:
            gamma1, beta1, alpha1, gamma3, beta3, alpha3 = mod.chunk(6, dim=-1)
        gamma1 = gamma1.unsqueeze(1); beta1 = beta1.unsqueeze(1); alpha1 = alpha1.unsqueeze(1)
        gamma3 = gamma3.unsqueeze(1); beta3 = beta3.unsqueeze(1); alpha3 = alpha3.unsqueeze(1)

        # Self-Attention
        h = self.norm_sa(x) * (1 + gamma1) + beta1
        x = x + alpha1 * self.self_attn(h, h, h)[0]

        # Cross-Attention (crossattn 모드에서만)
        if self.use_cross_attn:
            q = self.norm_ca_q(x)
            kv = self.norm_ca_kv(y_seq)
            x = x + alpha2 * self.cross_attn(q, kv, kv)[0]

        # FFN
        h = self.norm_ffn(x) * (1 + gamma3) + beta3
        x = x + alpha3 * self.ffn(h)
        return x


class Stable3DiT(nn.Module):
    """SD3/MMDiT-style Flow Matching DiT. adaLN-Zero(t) + cross-attention(Y)."""
    def __init__(
        self,
        num_points=201, num_bits=100, d_model=512, nhead=4,
        num_layers=15, dim_feedforward=768, dropout=0.1,
        seq_len_y=50, drop_prob=0.1, vocab_size=2, group_size=1,
        cond_mode="crossattn", mask_token=False,
    ):
        super().__init__()
        self.num_bits = num_bits
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.vocab_size = vocab_size
        self.group_size = group_size
        self.mask_token = mask_token
        # absorbing(mask) 또는 categorical(V>2) 이면 embedding 경로.
        # 입력 vocab 은 mask_token 이면 +1 (MASK id = vocab_size), 출력 헤드는 vocab_size 그대로.
        self.use_embed = (vocab_size > 2) or mask_token
        assert cond_mode in ("crossattn", "adaln")
        self.cond_mode = cond_mode
        use_cross = (cond_mode == "crossattn")

        if self.use_embed:
            self.input_embed = nn.Embedding(vocab_size + (1 if mask_token else 0), d_model)
        elif group_size > 1:
            self.input_proj = nn.Linear(group_size, d_model)
        else:
            self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Embedding(num_bits, d_model)
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )
        self.y_encoder = ResNet1DSeqEncoder(d_model, num_blocks=3, kernel_size=3, seq_len_y=seq_len_y)
        self.null_y_embed = nn.Parameter(torch.randn(1, seq_len_y, d_model) * 0.02)

        # adaln 모드: pooled spectrum 을 t_emb 에 더해 adaLN 을 (t+y) 로 구동
        if cond_mode == "adaln":
            self.y_pool_proj = nn.Linear(d_model, d_model)

        self.blocks = nn.ModuleList([
            Stable3DiTBlock(d_model, nhead, dim_feedforward, dropout, use_cross_attn=use_cross)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        if self.use_embed:
            out_dim = vocab_size          # 헤드는 실토큰만 (MASK 제외)
        elif group_size > 1:
            out_dim = group_size
        else:
            out_dim = 1
        self.output_proj = nn.Linear(d_model, out_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def _forward_core(self, x_t, t, y_seq):
        B = x_t.size(0); device = x_t.device
        if self.use_embed:
            x = self.input_embed(x_t.long())
        elif self.group_size > 1:
            x = self.input_proj(x_t)
        else:
            x = self.input_proj(x_t.unsqueeze(-1))
        x = x + self.pos_embed(torch.arange(self.num_bits, device=device)).unsqueeze(0)
        t_emb = self.t_mlp(timestep_embedding(t, self.d_model))

        # adaLN 구동 벡터: crossattn=t only, adaln=t + pooled(y_seq)
        if self.cond_mode == "adaln":
            cond = t_emb + self.y_pool_proj(y_seq.mean(dim=1))
            for block in self.blocks:
                x = block(x, cond, None)
        else:
            cond = t_emb
            for block in self.blocks:
                x = block(x, cond, y_seq)

        mod = self.final_adaLN(cond)
        gamma, beta = mod.chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        out = self.output_proj(x)
        if not self.use_embed and self.group_size <= 1:
            return out.squeeze(-1)
        return out

    def forward(self, x_t, t, y):
        B = x_t.size(0); device = x_t.device
        y_seq = self.y_encoder(y.unsqueeze(1))
        if self.training and self.drop_prob > 0:
            mask = (torch.rand(B, device=device) < self.drop_prob).view(B, 1, 1)
            y_seq = torch.where(mask, self.null_y_embed.expand(B, -1, -1), y_seq)
        return self._forward_core(x_t, t, y_seq)

    def forward_uncond(self, x_t, t):
        B = x_t.size(0)
        return self._forward_core(x_t, t, self.null_y_embed.expand(B, -1, -1))


# ===========================================================================
# SmallDiT: AR-comparable (additive conditioning, standard Transformer, no adaLN)
# ===========================================================================
class SmallDiT(nn.Module):
    """
    Minimal bidirectional Transformer for Flow Matching.
    AR Transformer와 동일 구조에서 causal mask만 제거.
    조건 주입: (t_emb + y_vec)를 input token에 additive — AR의 cond 방식과 동일.
    """
    def __init__(
        self,
        num_points=201, num_bits=100, d_model=512, nhead=4,
        num_layers=15, dim_feedforward=768, dropout=0.1,
        drop_prob=0.1, vocab_size=2, group_size=1, mask_token=False,
    ):
        super().__init__()
        self.num_bits = num_bits
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.vocab_size = vocab_size
        self.group_size = group_size
        self.mask_token = mask_token
        self.use_embed = (vocab_size > 2) or mask_token

        # Input embedding (absorbing 이면 MASK id = vocab_size 를 위해 +1 슬롯)
        if self.use_embed:
            self.input_embed = nn.Embedding(vocab_size + (1 if mask_token else 0), d_model)
        elif group_size > 1:
            self.input_proj = nn.Linear(group_size, d_model)
        else:
            self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Embedding(num_bits, d_model)

        # Time embedding
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )

        # Spectral condition encoder (global avg pool → vector)
        self.y_encoder = ResNet1DEncoder(d_model=d_model, num_blocks=3, kernel_size=3)
        self.null_y_embed = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # Standard TransformerEncoder (no causal mask, no adaLN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output (헤드는 실토큰만; MASK 는 입력에만 존재)
        self.output_norm = nn.LayerNorm(d_model)
        if self.use_embed:
            out_dim = vocab_size
        elif group_size > 1:
            out_dim = group_size
        else:
            out_dim = 1
        self.output_proj = nn.Linear(d_model, out_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _forward_core(self, x_t, t, y_vec):
        B = x_t.size(0); device = x_t.device

        # Input embedding
        if self.use_embed:
            x = self.input_embed(x_t.long())                         # (B, T, d_model)
        elif self.group_size > 1:
            x = self.input_proj(x_t)                                  # (B, T, gs) → (B, T, d_model)
        else:
            x = self.input_proj(x_t.unsqueeze(-1))                   # (B, T, 1) → (B, T, d_model)
        x = x + self.pos_embed(torch.arange(self.num_bits, device=device)).unsqueeze(0)

        # Condition: t_emb + y_vec, broadcast to all positions
        t_emb = self.t_mlp(timestep_embedding(t, self.d_model))      # (B, d_model)
        cond = (t_emb + y_vec).unsqueeze(1)                           # (B, 1, d_model)
        x = x + cond

        # Bidirectional Transformer (no causal mask)
        x = self.transformer(x)

        # Output
        x = self.output_norm(x)
        out = self.output_proj(x)                                     # (B, T, out_dim)
        if not self.use_embed and self.group_size <= 1:
            return out.squeeze(-1)                                    # (B, T)
        return out                                                    # (B, T, V or gs)

    def forward(self, x_t, t, y):
        B = x_t.size(0); device = x_t.device
        y_vec = self.y_encoder(y.unsqueeze(1))                        # (B, d_model)
        if self.training and self.drop_prob > 0:
            mask = (torch.rand(B, device=device) < self.drop_prob).view(B, 1)
            y_vec = torch.where(mask, self.null_y_embed.expand(B, -1), y_vec)
        return self._forward_core(x_t, t, y_vec)

    def forward_uncond(self, x_t, t):
        B = x_t.size(0)
        return self._forward_core(x_t, t, self.null_y_embed.expand(B, -1))


# ===========================================================================
# Factory: build model by name
# ===========================================================================
def build_model(arch, **kwargs):
    if arch == "stable3dit":
        return Stable3DiT(**kwargs)
    elif arch == "smalldit":
        # SmallDiT doesn't use seq_len_y / cross_attn / cond_mode
        kwargs.pop("seq_len_y", None)
        kwargs.pop("use_cross_attn", None)
        kwargs.pop("cond_mode", None)
        return SmallDiT(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
