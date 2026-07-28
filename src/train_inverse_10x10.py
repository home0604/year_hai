
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
import random

BOS_IDX = 2   # 0,1: 실제 비트, 2: BOS 토큰


# ---------------------------
# Seed 고정
# ---------------------------
def set_seed(seed: int):
    print(f"🔒 Setting global seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 더 deterministic하게 만들기 (약간 느려질 수 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# (옵션) HxW 복원: 디버깅/시각화용
# ---------------------------
def restore_pixel_order(X_flat, height, width):
    """
    입력이 (N, H*W) flatten되어 있을 때
    1xHxW로 복원 (row-major 기준).
    """
    N, num_bits = X_flat.shape
    assert num_bits == height * width, f"num_bits({num_bits}) != H*W({height*width})"
    X_restored = X_flat.reshape(N, 1, height, width)
    return X_restored.astype(np.float32)


# ---------------------------
# Canonicalization for y-axis symmetry (좌우 대칭)
# ---------------------------
def horizontal_flip_y_axis(X_flat, height, width):
    """
    HxW 패턴을 y축(좌우) 대칭으로 플립.
    X_flat: (N, H*W), row-major (0..H*W-1) ordering 가정.
    """
    N, num_bits = X_flat.shape
    assert num_bits == height * width
    X_reshaped = X_flat.reshape(N, height, width)
    X_flipped = X_reshaped[:, :, ::-1]   # 열 방향 반전
    return X_flipped.reshape(N, num_bits)


def canonicalize_under_yflip(X_flat, height, width):
    """
    각 패턴에 대해 y축 대칭 패턴을 만들고,
    두 벡터 중 사전순(lexicographic)으로 작은 쪽을 canonical 정답으로 사용.
    """
    X_flat = X_flat.copy()
    X_flip = horizontal_flip_y_axis(X_flat, height, width)
    N = X_flat.shape[0]
    X_can = np.empty_like(X_flat)

    for i in range(N):
        a = X_flat[i]
        b = X_flip[i]
        if list(a) <= list(b):
            X_can[i] = a
        else:
            X_can[i] = b
    return X_can


# ---------------------------
# Hilbert curve utilities (general 2^k x 2^k)
# ---------------------------
def _hilbert_rot(n, x, y, rx, ry):
    """
    Hilbert curve rotation helper.
    n: 현재 단계의 사이드 길이 (2^k)
    """
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        # swap x, y
        x, y = y, x
    return x, y


def _hilbert_d2xy(n, d):
    """
    distance d (0 <= d < n*n)를
    Hilbert curve 상의 (x, y) 좌표로 변환.
    n 은 2^k 여야 함.
    """
    x = 0
    y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _hilbert_rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def get_order_indices(ordering, num_bits, height, width):
    """
    HxW grid 상에서의 순회 순서를 정의.
    반환: order_idx (np.array of shape (num_bits,))
      - order_idx[t] = original flattened index at AR position t
    ordering:
      - 'raster': row-major 0..(H*W-1)
      - 'snake' : row별로 L-R, R-L 번갈아
      - 'hilbert': 2^k x 2^k 격자에 Hilbert curve를 정의한 뒤
                   (x < width, y < height)인 점만 순서대로 사용
    """
    assert num_bits == height * width, "num_bits must be H*W"

    if ordering == "raster":
        order = np.arange(num_bits, dtype=np.int64)

    elif ordering == "snake":
        order = []
        for r in range(height):
            if r % 2 == 0:
                cols = range(width)
            else:
                cols = reversed(range(width))
            for c in cols:
                idx = r * width + c
                order.append(idx)
        order = np.array(order, dtype=np.int64)

    elif ordering == "hilbert":
        # height, width 가 2^k가 아니어도 되도록
        # max(height, width)를 포함하는 최소 2^k 크기의 격자에서
        # Hilbert curve 를 생성한 뒤, (x < width, y < height)만 필터
        max_side = max(height, width)
        n_side = 1
        while n_side < max_side:
            n_side *= 2   # 2^k >= max_side

        coords = []
        for d in range(n_side * n_side):
            x, y = _hilbert_d2xy(n_side, d)  # (x,y) in [0, n_side)
            if x < width and y < height:
                coords.append((x, y))

            if len(coords) == num_bits:
                break

        assert len(coords) == num_bits, \
            f"Hilbert coords length {len(coords)} != num_bits {num_bits}"

        # flat index = y * width + x (row-major)
        order = np.array([y * width + x for (x, y) in coords], dtype=np.int64)

    else:
        raise ValueError(f"Unknown ordering: {ordering}")

    assert len(order) == num_bits
    return order


# ---------------------------
# 1D ResNet encoder for spectral condition
# ---------------------------
class ResNet1DEncoder(nn.Module):
    """
    y: (B, P) → (B, 1, P) → 1D ResNet → global average pooling → (B, d_model)
    P(=num_points)는 201까지/그 이상도 자유롭게 처리 가능.
    """
    def __init__(self, d_model, num_blocks=3, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.input_proj = nn.Conv1d(1, d_model, kernel_size=kernel_size, padding=padding)
        self.input_bn   = nn.BatchNorm1d(d_model)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(d_model),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(d_model),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.act = nn.ReLU(inplace=True)

    def forward(self, y):
        """
        y: (B, 1, P)
        """
        x = self.input_proj(y)        # (B, d_model, P)
        x = self.input_bn(x)
        x = self.act(x)

        for block in self.blocks:
            residual = x
            out = block(x)
            x = self.act(out + residual)

        # global average pooling over length dimension
        x = x.mean(dim=-1)           # (B, d_model)
        return x


# ---------------------------
# 작은 Transformer AR 모델
# ---------------------------
class SmallTransformerAR(nn.Module):
    """
    Autoregressive Transformer:
      - 입력 토큰: [BOS, x0, x1, ..., x_{T-1}] (길이 T = num_bits)
      - 출력: 각 위치 t에서 '다음 비트 x_t'에 대한 logit (길이 T)
        logits[:,0] -> x0, logits[:,1] -> x1, ..., logits[:,T-1] -> x_{T-1}
      - 조건: y (S11 스펙트럼) → cond_encoder → d_model → 모든 토큰 embedding에 더함
      - causal mask 사용 (i 위치는 <= i만 attend)
      - spectral_cond:
          * 'linear'     : Linear(num_points → d_model)
          * 'mlp'        : 2-layer MLP
          * 'transformer': small Transformer encoder over freq dimension
          * 'resnet1d'   : 1D ResNet encoder over freq dimension
      - use_2d_pos:
          * True  → 1D pos_embed + 2D pos2d_embed(chain→spatial index 기반) 둘 다 사용
          * False → 기존 1D pos_embed만 사용
    """
    def __init__(
        self,
        num_points,
        d_model=256,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        max_len=100,      # = num_bits
        vocab_size=3,    # 0,1,BOS
        dropout=0.1,
        spectral_cond="linear",   # 'linear' or 'mlp' or 'transformer' or 'resnet1d'
        use_2d_pos=False,
        chain2spatial=None,       # Tensor of shape (max_len,), chain index -> spatial index (0..H*W-1)
        height=10,
        width=10,
    ):
        super().__init__()
        self.num_points = num_points
        self.d_model = d_model
        self.max_len = max_len
        self.spectral_cond_type = spectral_cond
        self.use_2d_pos = use_2d_pos
        self.height = height
        self.width = width

        # 토큰/포지션/컨디션 임베딩
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(max_len, d_model)   # 1D (chain index)

        # 2D positional embedding (optional)
        if self.use_2d_pos:
            assert chain2spatial is not None, "use_2d_pos=True 이면 chain2spatial을 넘겨줘야 합니다."
            assert chain2spatial.numel() >= max_len
            # 0..(H*W-1) spatial index용 embedding
            self.pos2d_embed = nn.Embedding(height * width, d_model)
            # chain position t → spatial index (original flatten index)
            self.register_buffer("chain2spatial", chain2spatial.clone())  # shape: (max_len,)

        # spectral condition encoder
        if spectral_cond == "linear":
            self.cond_encoder = nn.Linear(num_points, d_model)
        elif spectral_cond == "mlp":
            self.cond_encoder = nn.Sequential(
                nn.Linear(num_points, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
        elif spectral_cond == "transformer":
            # y: (B, num_points) → (B, num_points, 1) → proj → +pos → TransformerEncoder → mean-pool
            self.freq_in_proj = nn.Linear(1, d_model)
            self.freq_pos_embed = nn.Embedding(num_points, d_model)
            cond_encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,                   # 작은 헤드 수
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.cond_transformer = nn.TransformerEncoder(
                cond_encoder_layer,
                num_layers=2
            )
        elif spectral_cond == "resnet1d":
            # y: (B, P) → (B,1,P) → ResNet1D → (B,d_model)
            self.resnet1d = ResNet1DEncoder(d_model=d_model, num_blocks=3, kernel_size=3)
        else:
            raise ValueError(f"Unknown spectral_cond: {spectral_cond}")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc_out = nn.Linear(d_model, 1)  # 각 위치의 logit

    def _generate_causal_mask(self, L, device):
        # (L,L) 상삼각 부분 True → 미래를 mask
        # TransformerEncoder: mask에서 True = "볼 수 없음"
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
        return mask

    def encode_condition(self, y):
        """
        y: (B, num_points)
        반환: cond_vec: (B, d_model)
        """
        if self.spectral_cond_type in ["linear", "mlp"]:
            return self.cond_encoder(y)  # (B,d_model)
        elif self.spectral_cond_type == "transformer":
            B, P = y.shape
            device = y.device
            y_seq = y.unsqueeze(-1)                 # (B,P,1)
            feat = self.freq_in_proj(y_seq)        # (B,P,d_model)

            pos_idx = torch.arange(P, device=device).unsqueeze(0).expand(B, P)
            pos_emb = self.freq_pos_embed(pos_idx) # (B,P,d_model)

            feat = feat + pos_emb                  # (B,P,d_model)
            h = self.cond_transformer(feat)        # (B,P,d_model)

            cond_vec = h.mean(dim=1)               # (B,d_model) global average pooling
            return cond_vec
        elif self.spectral_cond_type == "resnet1d":
            # y: (B,P) -> (B,1,P)
            y_seq = y.unsqueeze(1)
            cond_vec = self.resnet1d(y_seq)        # (B,d_model)
            return cond_vec
        else:
            raise RuntimeError("Invalid spectral_cond_type")

    def forward(self, y, tokens):
        """
        y:      (B, num_points)
        tokens: (B, L<=max_len)  [BOS, x0, ..., x_{L-2}]
        반환:   logits: (B, L), 각 위치의 '다음 비트' logit
        """
        B, L = tokens.shape
        device = tokens.device

        tok_emb = self.token_embed(tokens)              # (B,L,d_model)

        # 1D positional embedding (chain index)
        pos_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B,L)
        pos_emb = self.pos_embed(pos_idx)               # (B,L,d_model)

        # 2D positional embedding (optional)
        if self.use_2d_pos:
            spatial_idx = self.chain2spatial[:L].to(device)          # (L,)
            spatial_idx = spatial_idx.unsqueeze(0).expand(B, L)      # (B,L)
            pos2d_emb = self.pos2d_embed(spatial_idx)                # (B,L,d_model)
            pos_emb = pos_emb + pos2d_emb

        cond_vec = self.encode_condition(y)             # (B,d_model)
        cond = cond_vec.unsqueeze(1).expand(B, L, self.d_model)  # (B,L,d_model)

        x = tok_emb + pos_emb + cond                    # (B,L,d_model)

        src_mask = self._generate_causal_mask(L, device=device)  # (L,L) bool

        h = self.transformer(x, mask=src_mask)          # (B,L,d_model)

        logits = self.fc_out(h).squeeze(-1)             # (B,L)
        return logits


# ---------------------------
# Train / Eval 루프
# ---------------------------
def train_one_epoch_transformer(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_bit_acc = 0.0
    total_pattern_acc = 0.0

    for yb, xb in loader:
        # yb: (B, num_points), xb: (B,num_bits) {0,1}  (이미 ordering 적용된 상태)
        yb = yb.to(device)
        xb = xb.to(device)
        B, num_bits = xb.shape

        # 입력 토큰: [BOS, x0, ..., x_{T-1}]  (길이 T=num_bits)
        bos = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
        x_prev = xb[:, :-1].long()                      # (B,T-1)
        tokens_in = torch.cat([bos, x_prev], dim=1)     # (B,T)

        targets = xb                                   # (B,T)

        optimizer.zero_grad()
        logits = model(yb, tokens_in)                  # (B,T)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        total_samples += B

        # accuracy
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()              # (B,T)

            bit_correct = (preds == targets).float().mean().item()
            pattern_correct = (preds == targets).all(dim=1).float().mean().item()

            total_bit_acc += bit_correct * B
            total_pattern_acc += pattern_correct * B

    avg_loss = total_loss / total_samples
    avg_bit_acc = total_bit_acc / total_samples
    avg_pattern_acc = total_pattern_acc / total_samples
    return avg_loss, avg_bit_acc, avg_pattern_acc


def eval_transformer(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_bit_acc = 0.0
    total_pattern_acc = 0.0

    with torch.no_grad():
        for yb, xb in loader:
            yb = yb.to(device)
            xb = xb.to(device)
            B, num_bits = xb.shape

            bos = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
            x_prev = xb[:, :-1].long()
            tokens_in = torch.cat([bos, x_prev], dim=1)  # (B,T)

            targets = xb                                  # (B,T)

            logits = model(yb, tokens_in)                 # (B,T)
            loss = criterion(logits, targets)

            total_loss += loss.item() * B
            total_samples += B

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            bit_correct = (preds == targets).float().mean().item()
            pattern_correct = (preds == targets).all(dim=1).float().mean().item()

            total_bit_acc += bit_correct * B
            total_pattern_acc += pattern_correct * B

    avg_loss = total_loss / total_samples
    avg_bit_acc = total_bit_acc / total_samples
    avg_pattern_acc = total_pattern_acc / total_samples
    return avg_loss, avg_bit_acc, avg_pattern_acc


# ---------------------------
# 샘플링: y 주어졌을 때 X 시퀀스를 AR로 생성
#   - 내부 체인 순서는 ordering 기준
#   - 반환은 "original flatten ordering" (0..H*W-1) 기준
# ---------------------------
def sample_transformer_ar(model, y, num_bits, greedy,
                          ordering, height, width):
    """
    y: (B, num_points)
    반환: 샘플링된 X_flat: (B,num_bits) in {0,1}, original flatten ordering
    """
    model.eval()
    device = next(model.parameters()).device
    y = y.to(device)
    B = y.size(0)

    # 시작 토큰: [BOS]
    tokens = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)

    bits_chain = []
    with torch.no_grad():
        for t in range(num_bits):
            # tokens: [BOS, x0, ..., x_{t-1}]  (길이 t+1 <= num_bits)
            logits = model(y, tokens)      # (B, t+1)
            next_logit = logits[:, -1]     # 마지막 위치 → x_t 예측

            probs = torch.sigmoid(next_logit)
            if greedy:
                x_t = (probs > 0.5).long()
            else:
                x_t = torch.bernoulli(probs).long()

            bits_chain.append(x_t.unsqueeze(1))  # (B,1)

            tokens = torch.cat([tokens, x_t.unsqueeze(1)], dim=1)  # prefix 확장

    bits_chain = torch.cat(bits_chain, dim=1)  # (B,num_bits) in chain-order

    # chain-order → original-order 로 역변환
    order_idx_np = get_order_indices(ordering, num_bits, height, width)
    order_idx = torch.from_numpy(order_idx_np).to(device)  # (num_bits,)
    bits_flat = torch.zeros_like(bits_chain)
    bits_flat[:, order_idx] = bits_chain

    return bits_flat  # original flatten ordering


# ---------------------------
# Top-k pattern accuracy via sampling
# ---------------------------
def eval_topk_pattern_accuracy(model, Y_test_t, X_test_flat, k, device,
                               ordering, num_bits, height, width):
    """
    각 test sample에 대해 k번 샘플링했을 때,
    그 중 하나라도 GT 패턴과 완전히 일치하면 correct로 카운트.
    (GT는 canonical_target 옵션에 의해 정규화 되었을 수 있음)
    """
    model.eval()
    N = Y_test_t.size(0)
    X_test_flat_t = torch.tensor(X_test_flat, device=device).float()
    correct = 0

    with torch.no_grad():
        for i in range(N):
            y_i = Y_test_t[i:i+1].to(device)          # (1,num_points)
            y_rep = y_i.repeat(k, 1)                  # (k,num_points)
            samples = sample_transformer_ar(
                model, y_rep,
                num_bits=num_bits,
                greedy=False,
                ordering=ordering,
                height=height,
                width=width,
            )                                         # (k,num_bits)

            gt = X_test_flat_t[i].unsqueeze(0).repeat(k, 1)  # (k,num_bits)
            match = (samples == gt).all(dim=1)               # (k,)
            if match.any():
                correct += 1

    topk_acc = correct / N
    return topk_acc


# ---------------------------
# Main
# ---------------------------
def main(args):
    # Seed 고정
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🟢 Using device: {device}")

    # --- Data load (새 10x10 형식) ---
    data_root = args.data_root
    train_npz = np.load(os.path.join(data_root, "dataset_train.npz"))
    valid_npz = np.load(os.path.join(data_root, "dataset_valid.npz"))
    test_npz  = np.load(os.path.join(data_root, "dataset_test.npz"))

    # X: (N,H,W), Y: (N,num_points)
    X_train_2d = train_npz["X"].astype(np.float32)
    Y_train    = train_npz["Y"].astype(np.float32)
    X_val_2d   = valid_npz["X"].astype(np.float32)
    Y_val      = valid_npz["Y"].astype(np.float32)
    X_test_2d  = test_npz["X"].astype(np.float32)
    Y_test     = test_npz["Y"].astype(np.float32)

    freq = train_npz["freq"].astype(np.float32)
    print("Using original resolution:", len(freq), "points")

    N_train, H, W = X_train_2d.shape
    num_points = Y_train.shape[1]
    num_bits   = H * W
    print(f"✅ Dataset loaded: train={len(X_train_2d)}, val={len(X_val_2d)}, test={len(X_test_2d)}, "
          f"H={H}, W={W}, num_bits={num_bits}, num_points={num_points}")

    # flatten (original ordering: row-major)
    X_train_flat = X_train_2d.reshape(N_train, num_bits)
    X_val_flat   = X_val_2d.reshape(X_val_2d.shape[0], num_bits)
    X_test_flat  = X_test_2d.reshape(X_test_2d.shape[0], num_bits)

    # --- canonical target (y축 대칭 포함) 옵션 ---
    if args.canonical_target:
        print("🔁 Applying canonicalization under y-axis flip to targets (X_train/X_val/X_test).")
        X_train_flat = canonicalize_under_yflip(X_train_flat, height=H, width=W)
        X_val_flat   = canonicalize_under_yflip(X_val_flat,   height=H, width=W)
        X_test_flat  = canonicalize_under_yflip(X_test_flat,  height=H, width=W)

    # --- ordering: original → chain-order 로 permute ---
    order_idx = get_order_indices(args.ordering, num_bits, H, W)  # (num_bits,)
    print(f"🔁 Using ordering = {args.ordering}, order_idx[:20] = {order_idx[:20].tolist()}")

    X_train_ord = X_train_flat[:, order_idx]
    X_val_ord   = X_val_flat[:, order_idx]
    X_test_ord  = X_test_flat[:, order_idx]

    # --- per-sample 입력 normalization (옵션) ---
    if args.normalize_input:
        # 각 sample에 대해 (freq dimension) 평균 0
        def norm_per_sample(Y):
            mean = Y.mean(axis=1, keepdims=True)
            return (Y - mean)

        Y_train = norm_per_sample(Y_train)
        Y_val   = norm_per_sample(Y_val)
        Y_test  = norm_per_sample(Y_test)
        print("🔧 Per-sample normalization applied to Y (train/val/test).")

    # torch tensor (주의: X_*_ord 사용)
    Y_train_t = torch.tensor(Y_train)
    X_train_t = torch.tensor(X_train_ord)
    Y_val_t   = torch.tensor(Y_val)
    X_val_t   = torch.tensor(X_val_ord)
    Y_test_t  = torch.tensor(Y_test)
    X_test_t  = torch.tensor(X_test_ord)

    train_loader = DataLoader(TensorDataset(Y_train_t, X_train_t),
                              batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Y_val_t,   X_val_t),
                              batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(Y_test_t,  X_test_t),
                              batch_size=args.batch_size, shuffle=False)

    # --- W&B init ---
    run_name = (
        f"{args.run_prefix}-"
        f"L{args.num_layers}-d{args.d_model}-h{args.nhead}-ff{args.dim_ff}-"
        f"dr{args.dropout}-ord{args.ordering}-spec{args.spectral_cond}-"
        f"2dpos{int(args.use_2d_pos)}-canon{int(args.canonical_target)}-"
        f"lr{args.lr}-bs{args.batch_size}-seed{args.seed}"
    )
    wandb.init(project=args.project, name=run_name)
    # 하이퍼파라미터 sweep-friendly: 모든 args를 config에 기록
    wandb.config.update(vars(args))

    # chain position → spatial index (original flat index) 텐서 (2D pos용)
    chain2spatial = torch.from_numpy(order_idx).long()

    # --- Model / Loss / Optim / LR Scheduler ---
    model = SmallTransformerAR(
        num_points=num_points,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        max_len=num_bits,   # =H*W
        vocab_size=3,
        dropout=args.dropout,
        spectral_cond=args.spectral_cond,
        use_2d_pos=args.use_2d_pos,
        chain2spatial=chain2spatial,
        height=H,
        width=W,
    ).to(device)

    print(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_scheduler == "cosine":
        # 5 epoch warm-up @ lr_warmup, 이후 cosine annealing from lr_max(=args.lr) to 0
        warmup_epochs = 5
        lr_warmup = 1e-4
        lr_max = args.lr
        eta_min = 0.0

        def lr_lambda(epoch):
            """
            epoch: 0-based index (LambdaLR 내부에서 사용)
            우리가 원하는 것은:
              - ep = 1..warmup_epochs: lr = lr_warmup
              - ep = warmup_epochs+1 .. args.epochs: cosine(lr_max -> eta_min)
            """
            ep = epoch + 1  # 1-based
            if ep <= warmup_epochs:
                return lr_warmup / lr_max
            # cosine 구간 길이
            T = max(args.epochs - warmup_epochs, 1)
            # t in [0,1]
            if T > 1:
                t = float(ep - warmup_epochs - 1) / float(T - 1)
            else:
                t = 0.0
            cos_factor = 0.5 * (1.0 + np.cos(np.pi * t))  # 1 -> 0
            # eta_min는 0이므로 cos_factor만 사용
            return cos_factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        print(f"🔄 Using warm-up + cosine LR schedule: warmup_epochs={warmup_epochs}, "
              f"lr_warmup={lr_warmup}, lr_max={lr_max}")

    os.makedirs(args.save_dir, exist_ok=True)
    # run별로 디렉토리 분리
    run_save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_save_dir, exist_ok=True)
    save_path = os.path.join(run_save_dir, "best_model.pth")

    best_val_loss = float("inf")

    # --- Train loop ---
    for epoch in range(1, args.epochs + 1):
        train_loss, train_bit_acc, train_pat_acc = train_one_epoch_transformer(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_bit_acc, val_pat_acc = eval_transformer(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"[Epoch {epoch:03d}/{args.epochs}] "
              f"Train Loss: {train_loss:.6f}, BitAcc: {train_bit_acc:.4f}, PatAcc: {train_pat_acc:.4f} | "
              f"Val Loss: {val_loss:.6f}, BitAcc: {val_bit_acc:.4f}, PatAcc: {val_pat_acc:.4f} | "
              f"lr={current_lr:.3e}")

        wandb.log({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_bit_acc": train_bit_acc,
            "train_pattern_acc": train_pat_acc,
            "val_loss": val_loss,
            "val_bit_acc": val_bit_acc,
            "val_pattern_acc": val_pat_acc,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  💾 Best model updated & saved to {save_path}")
            wandb.run.summary["best_val_loss"] = best_val_loss

        if scheduler is not None:
            scheduler.step()

    # --- Test ---
    print("🔍 Evaluating best model on test set...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_bit_acc, test_pat_acc = eval_transformer(
        model, test_loader, criterion, device
    )
    print(f"📊 Test Loss: {test_loss:.6f}, Test BitAcc: {test_bit_acc:.4f}, Test PatAcc: {test_pat_acc:.4f}")

    wandb.log({
        "test_loss": test_loss,
        "test_bit_acc": test_bit_acc,
        "test_pattern_acc": test_pat_acc,
    })
    wandb.run.summary["test_loss"] = test_loss
    wandb.run.summary["test_bit_acc"] = test_bit_acc
    wandb.run.summary["test_pattern_acc"] = test_pat_acc

    # --- Top-k pattern accuracy via sampling ---
    print(f"🎯 Evaluating Top-{args.topk} pattern accuracy via sampling...")
    topk_acc = eval_topk_pattern_accuracy(
        model, Y_test_t, X_test_flat, k=args.topk, device=device,
        ordering=args.ordering, num_bits=num_bits, height=H, width=W
    )
    print(f"📈 Top-{args.topk} Pattern Accuracy: {topk_acc:.4f}")
    wandb.log({f"top{args.topk}_pattern_acc": topk_acc})
    wandb.run.summary[f"top{args.topk}_pattern_acc"] = topk_acc

    # --- 샘플링 예시 + GT/PRED 비교 이미지 업로드 ---
    model.eval()
    with torch.no_grad():
        num_samples = min(32, len(X_test_flat))   # 이미지로 올릴 샘플 수
        idx = torch.randperm(len(X_test_flat))[:num_samples]
        Y_sample = Y_test_t[idx].to(device)
        X_true   = X_test_flat[idx.numpy()]              # original ordering (flatten, canonicalized일 수 있음)
        X_gen    = sample_transformer_ar(
            model, Y_sample, num_bits=num_bits,
            greedy=False, ordering=args.ordering,
            height=H, width=W
        ).cpu().numpy()                                  # original ordering (flatten)

    # HxW로 복원
    X_true_img = restore_pixel_order(X_true, H, W)  # (N,1,H,W)
    X_gen_img  = restore_pixel_order(X_gen,  H, W)

    images = []
    for i in range(num_samples):
        fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        axes[0].imshow(X_true_img[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("GT")
        axes[0].axis("off")

        axes[1].imshow(X_gen_img[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Pred")
        axes[1].axis("off")

        plt.tight_layout()
        images.append(wandb.Image(fig, caption=f"Sample #{int(idx[i])}"))
        plt.close(fig)

    wandb.log({"GT_vs_Pred_Layouts": images})

    print("👉 Example samples (first few, flatten, original ordering):")
    print("True:")
    print(X_true[:8])
    print("Generated (sampled):")
    print(X_gen[:8])

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", type=str,
        default=os.path.join("../data/year/datasets/pa10x10/60_density_500000_dataset/")
    )
    parser.add_argument("--save_dir", type=str, default="./transformer_ar_inverse_models")

    # Transformer 하이퍼파라미터
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_ff", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)

    # ordering / conditioning / 2D pos 옵션
    parser.add_argument("--ordering", type=str, default="hilbert",
                        choices=["raster", "snake", "hilbert"])
    parser.add_argument("--spectral_cond", type=str, default="resnet1d",
                        choices=["linear", "mlp", "transformer", "resnet1d"])
    parser.add_argument("--use_2d_pos", type=lambda x: x.lower()=="true",
                        default=False,
                        help="Use 2D positional encoding (True/False)")

    # canonical target 옵션 (y축 대칭까지 같은 정답으로 간주하고, canonical 정답만 학습)
    parser.add_argument("--canonical_target", action="store_true",
                        help="If set, map each HxW pattern to a canonical representative "
                             "under y-axis flip using lexicographic order.")

    # 학습 하이퍼파라미터
    parser.add_argument("--lr", type=float, default=5e-4)  # cosine 쓸 땐 0.001 권장 (warmup 0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=["none", "cosine"])

    # top-k evaluation
    parser.add_argument("--topk", type=int, default=10,
                        help="k for top-k pattern accuracy evaluation via sampling")

    # 입력 normalization 옵션
    parser.add_argument("--normalize_input", action="store_true",
                        help="If set, per-sample normalize Y (zero mean).")

    # Seed 옵션
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for reproducibility.")

    # W&B 옵션
    parser.add_argument("--project", type=str, default="rogers_inverse_ar")
    parser.add_argument("--run_prefix", type=str, default="ar10x10")

    args = parser.parse_args()
    main(args)


# ar10x10-L9-d192-h4-ff768-dr0.1-ordhilbert-specresnet1d-2dpos0-canon1-lr0.0005-bs128-seed42 (don't forget canonical target) (77.182)

