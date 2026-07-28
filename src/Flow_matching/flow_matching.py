# -*- coding: utf-8 -*-
"""
Flow Matching for binary inverse design.

Two formulations:
  1. CFM (Continuous Flow Matching):
     - 이진 패턴 {0,1} → 연속 relaxation {-1,+1}
     - 직선 경로 보간, MSE velocity regression
     - Euler ODE 적분 → threshold

  2. DFM (Discrete Flow Matching):
     - 이산 상태 {0,1} 그대로 유지
     - 확률 보간: 각 bit를 확률 t로 데이터 유지, (1-t)로 random flip
     - 모델이 x₁ (clean data)의 logit을 예측 → BCE loss
     - 샘플링: 예측 확률에 따라 stochastic bit update
"""
import torch
import torch.nn.functional as F


# ===========================================================================
# Shared utilities
# ===========================================================================

def bits01_to_pm1(x01):
    """{0,1} -> {-1,+1}"""
    return x01 * 2.0 - 1.0


def pm1_to_bits01(xpm1):
    """{-1,+1} 연속값 -> 0 기준 threshold -> {0,1} long"""
    return (xpm1 > 0).long()


# ===========================================================================
# CFM: Continuous Flow Matching
# ===========================================================================

def cfm_loss(model, y, x1, sigma_min=0.0, group_size=1):
    """
    CFM loss. group_size>1이면 bits를 group해서 multi-dim continuous로 처리.
    x1: (B, num_bits) in {-1,+1}
    """
    B = x1.size(0)
    device = x1.device

    if group_size > 1:
        x1 = x1.reshape(B, -1, group_size)           # (B, T, gs)

    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device)
    t_b = t.view(B, *([1] * (x1.dim() - 1)))

    x_t = (1 - t_b) * x0 + t_b * x1
    if sigma_min > 0:
        x_t = x_t + sigma_min * torch.randn_like(x_t)
    u_t = x1 - x0

    v = model(x_t, t, y)
    return ((v - u_t) ** 2).mean()


@torch.no_grad()
def sample_fm(model, y, num_bits, steps=50, return_continuous=False):
    """
    y: (B, num_points)
    Euler ODE integration t: 0 -> 1, 그 후 thresholding.
    return: bits (B, num_bits) in {0,1} long  (return_continuous=True면 연속값도 반환)
    """
    model.eval()
    device = next(model.parameters()).device
    y = y.to(device)
    B = y.size(0)

    x = torch.randn(B, num_bits, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((B,), i * dt, device=device)
        v = model(x, t, y)
        x = x + v * dt

    bits = pm1_to_bits01(x)
    if return_continuous:
        return bits, x
    return bits


@torch.no_grad()
def sample_fm_cfg(model, y, num_bits, steps=50, guidance_scale=1.0,
                  return_continuous=False, group_size=1):
    """CFM sampling with CFG. group_size>1이면 grouped continuous."""
    model.eval()
    device = next(model.parameters()).device
    y = y.to(device)
    B = y.size(0)

    if group_size > 1:
        num_tokens = num_bits // group_size
        x = torch.randn(B, num_tokens, group_size, device=device)
    else:
        x = torch.randn(B, num_bits, device=device)

    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((B,), i * dt, device=device)
        v = _cfg_logits(model, x, t, y, guidance_scale)
        x = x + v * dt

    if group_size > 1:
        x = x.reshape(B, -1)                         # (B, num_bits)
    bits = pm1_to_bits01(x)
    if return_continuous:
        return bits, x
    return bits


# ===========================================================================
# CFM x-prediction + STE (discretization-aware regression)
# ===========================================================================

def cfm_xpred_ste_loss(model, y, x1, lambda_ste=0.5, ste_t_weight=True,
                       group_size=1):
    """CFM인데 모델 출력을 velocity 가 아니라 x̂₁ (x-prediction)로 해석한다.

    목적함수 (둘 다 회귀 → CFM 프레임 유지):
      (a) 연속 회귀   : ||x̂₁ - x₁||²          흐름이 올바른 x₁ 를 향하게
      (b) STE 이산화 : ||round(x̂₁) - x₁||²    부호 틀린 비트에만, grad 는 항등 통과

    round 는 추론 threshold(pm1_to_bits01: x>0 → 1)와 정확히 일치시킨다.
    x1: (B, num_bits) in {-1,+1}
    """
    B = x1.size(0)
    x1_bits = x1                                       # (B, D) — STE 용 원본
    if group_size > 1:
        x1 = x1.reshape(B, -1, group_size)

    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=x1.device)
    t_b = t.view(B, *([1] * (x1.dim() - 1)))
    x_t = (1 - t_b) * x0 + t_b * x1

    x1_hat = model(x_t, t, y)                          # ← x̂₁ (velocity 아님)

    # (a) 연속 회귀 — 흐름 본체
    loss_reg = ((x1_hat - x1) ** 2).mean()

    # (b) STE 이산화 — 값은 이산, 기울기는 항등 통과
    xh = x1_hat.reshape(B, -1)
    b = torch.where(xh > 0, 1.0, -1.0)                 # 추론 threshold 와 동일
    b_ste = xh + (b - xh).detach()                     # forward=b, backward=identity
    per_bit = (b_ste - x1_bits.reshape(B, -1)) ** 2    # 틀린 비트만 4, 맞으면 0
    if ste_t_weight:
        per_bit = per_bit * t.view(B, 1)               # 작은 t(노이즈 지배) 다운웨이트
    loss_ste = per_bit.mean()

    return loss_reg + lambda_ste * loss_ste


@torch.no_grad()
def sample_fm_xpred_cfg(model, y, num_bits, steps=50, guidance_scale=1.0,
                        return_continuous=False, group_size=1):
    """x-prediction CFM 샘플러: 모델 출력=x̂₁, v=(x̂₁-x_t)/(1-t).

    마지막 스텝(t=1-dt)에서 (1-t)=dt 라 x 가 정확히 x̂₁ 에 착지 → 나눗셈 폭발 없음.
    """
    model.eval()
    device = next(model.parameters()).device
    y = y.to(device)
    B = y.size(0)

    if group_size > 1:
        x = torch.randn(B, num_bits // group_size, group_size, device=device)
    else:
        x = torch.randn(B, num_bits, device=device)

    dt = 1.0 / steps
    for i in range(steps):
        t = i * dt
        t_b = torch.full((B,), t, device=device)
        x1_hat = _cfg_logits(model, x, t_b, y, guidance_scale)   # CFG 는 x̂₁ 에 선형
        v = (x1_hat - x) / max(1.0 - t, dt)                      # t<1 보장 + 안전 clamp
        x = x + v * dt

    if group_size > 1:
        x = x.reshape(B, -1)
    bits = pm1_to_bits01(x)
    if return_continuous:
        return bits, x
    return bits


# ===========================================================================
# Bit grouping utilities (for DFM with larger vocab)
# ===========================================================================

def make_spatial_patch_indices(H, W, ph, pw):
    """2D spatial patch 순서로 bits를 재배열하는 인덱스 생성.
    Returns: (fwd_idx, inv_idx) — fwd로 raster→patch, inv로 patch→raster.
    """
    indices = []
    for i in range(0, H, ph):
        for j in range(0, W, pw):
            for di in range(ph):
                for dj in range(pw):
                    indices.append((i + di) * W + (j + dj))
    fwd = torch.tensor(indices, dtype=torch.long)
    inv = torch.empty_like(fwd)
    inv[fwd] = torch.arange(len(fwd))
    return fwd, inv


def bits_to_tokens(x_bits, group_size):
    """(B, num_bits) {0,1} → (B, num_tokens) {0,...,2^group_size - 1}"""
    B, D = x_bits.shape
    x = x_bits.long().reshape(B, D // group_size, group_size)
    powers = (2 ** torch.arange(group_size, device=x_bits.device))
    return (x * powers).sum(-1)


def tokens_to_bits(tokens, group_size):
    """(B, num_tokens) {0,...,2^group_size - 1} → (B, num_bits) {0,1}"""
    B, T = tokens.shape
    bits = []
    t = tokens
    for _ in range(group_size):
        bits.append(t & 1)
        t = t >> 1
    return torch.stack(bits, dim=-1).reshape(B, T * group_size).float()


# ===========================================================================
# DFM: Discrete Flow Matching
# ===========================================================================

def _corrupt_bits(x1, t):
    """
    Discrete noising (independent coupling):
    각 bit를 확률 t로 데이터 값 유지, (1-t)로 uniform random bit.
    x1: (B, num_bits) in {0, 1} float
    t:  (B,) in [0, 1]
    return: x_t (B, num_bits) in {0, 1} float
    """
    B, D = x1.shape
    device = x1.device
    t_b = t.view(B, 1)
    keep_mask = torch.rand(B, D, device=device) < t_b     # True면 데이터 유지
    random_bits = torch.randint(0, 2, (B, D), device=device).float()
    x_t = torch.where(keep_mask, x1, random_bits)
    return x_t


def _ot_corrupt_bits(x1, t):
    """
    Discrete noising with mini-batch OT coupling (exact Hungarian).
    x₀ ~ Uniform({0,1}^D) 를 x₁과 Hamming distance 기준 OT 매칭.
    ~1ms/batch overhead (B=128, D=100).
    """
    from scipy.optimize import linear_sum_assignment

    B, D = x1.shape
    device = x1.device

    x0 = torch.randint(0, 2, (B, D), device=device).float()

    with torch.no_grad():
        cost = (x0.sum(-1, keepdim=True) + x1.sum(-1).unsqueeze(0)
                - 2.0 * x0 @ x1.T).cpu().numpy()
    _, col_ind = linear_sum_assignment(cost)
    inv_perm = torch.empty(B, dtype=torch.long)
    inv_perm[torch.from_numpy(col_ind)] = torch.arange(B)
    x0_matched = x0[inv_perm.to(device)]

    t_b = t.view(B, 1)
    keep_mask = torch.rand(B, D, device=device) < t_b
    x_t = torch.where(keep_mask, x1, x0_matched)
    return x_t


def _corrupt_tokens(x1, t, vocab_size):
    """Token-level corruption: keep with prob t, uniform random with prob (1-t)."""
    B, T = x1.shape
    device = x1.device
    t_b = t.view(B, 1)
    keep_mask = torch.rand(B, T, device=device) < t_b
    random_tokens = torch.randint(0, vocab_size, (B, T), device=device)
    return torch.where(keep_mask, x1, random_tokens)


def _apply_flip_residual(logits, x_t, vocab_size, t):
    """Add identity prior: model output = correction from keeping x_t.
    Residual strength scales with t (higher t → more tokens already correct)."""
    B = t.size(0)
    keep_logit = F.one_hot(x_t.long(), vocab_size).float()
    alpha = t.view(B, 1, 1) * 5.0
    return logits + keep_logit * alpha


def dfm_loss(model, y, x1, ot=False, group_size=1, flip_pred=False):
    """
    Discrete Flow Matching loss.
    group_size=1: binary (V=2), BCE loss
    group_size>1: grouped tokens (V=2^gs), cross-entropy loss
    flip_pred: residual logit parameterization (model predicts correction from x_t)
    """
    B = x1.size(0)
    device = x1.device
    t = torch.rand(B, device=device)

    if group_size <= 1:
        corrupt_fn = _ot_corrupt_bits if ot else _corrupt_bits
        x_t = corrupt_fn(x1, t)
        x_t_input = x_t * 2.0 - 1.0
        logits = model(x_t_input, t, y)
        return F.binary_cross_entropy_with_logits(logits, x1)
    else:
        vocab_size = 2 ** group_size
        tokens = bits_to_tokens(x1, group_size)
        tokens_t = _corrupt_tokens(tokens, t, vocab_size)
        logits = model(tokens_t, t, y)              # (B, num_tokens, V)
        if flip_pred:
            logits = _apply_flip_residual(logits, tokens_t, vocab_size, t)
        return F.cross_entropy(logits.reshape(-1, vocab_size), tokens.reshape(-1))


import math


def _keep_schedule(steps, schedule="linear"):
    """스텝별 (keep-level α, per-token 점프율 a) 를 반환.

    학습 corruption 은 각 심볼을 확률 κ_t 로 유지한다 (keep_mask = rand < t).
    모델의 `t` 입력이 곧 이 keep-level 이므로, 샘플러는 매 스텝
      · 모델에 α_i = κ(τ_i) 를 넣고,
      · 점프율 a_i = (κ(τ_{i+1}) − κ(τ_i)) / (1 − κ(τ_i)) 로 예측 쪽으로 점프한다.
    이렇게 두면 corruption(κ)과 샘플러가 항상 정합이라 스케줄을 자유 노브로 쓸 수 있다.

    schedule: 'linear' | 'cosine' | callable(τ∈[0,1] → α∈[0,1], κ(0)=0·κ(1)=1·단조증가)

    linear(κ_t=t)·균등스텝에서는 α_i = i/steps, a_i = 1/(steps−i) 로 기존과 정확히 동일.
    """
    if schedule == "linear":
        kap = lambda u: u
    elif schedule == "cosine":
        kap = lambda u: math.sin(0.5 * math.pi * u) ** 2
    elif callable(schedule):
        kap = schedule
    else:
        raise ValueError(f"unknown schedule: {schedule!r}")

    alphas, jumps = [], []
    for i in range(steps):
        a_i = float(kap(i / steps))
        a_next = float(kap((i + 1) / steps))          # τ_{steps} = 1 → κ(1)=1
        alphas.append(a_i)
        denom = 1.0 - a_i
        jumps.append(1.0 if denom <= 1e-8 else min(1.0, (a_next - a_i) / denom))
    return alphas, jumps


@torch.no_grad()
def sample_dfm_cfg(model, y, num_bits, steps=50, guidance_scale=1.0,
                   group_size=1, flip_pred=False, schedule="linear", temperature=1.0):
    """DFM sampling with CFG. group_size>1이면 token-level categorical sampling.

    schedule: keep-level κ_t 스케줄 ('linear' 기본; 'cosine' 또는 callable).
              corruption 과 정합인 일반 점프율을 쓰므로 재학습 없이 교체 가능.
    """
    model.eval()
    device = next(model.parameters()).device
    y = y.to(device)
    B = y.size(0)
    alphas, jumps = _keep_schedule(steps, schedule)

    if group_size <= 1:
        # --- Binary sampling: exact finite-step transition (state-preserving) ---
        # per-bit: prob a_i = (κ_{t'}-κ_t)/(1-κ_t) 로 예측 posterior에서 재샘플, 아니면 x_t 유지.
        # 모델에는 keep-level α_i = κ_t 를 넣는다 (학습의 t 입력과 정합).
        x = torch.randint(0, 2, (B, num_bits), device=device).float()
        for i in range(steps):
            t = torch.full((B,), alphas[i], device=device)
            logits = _cfg_logits(model, x * 2.0 - 1.0, t, y, guidance_scale)
            p1 = torch.sigmoid(logits / temperature)         # P(x1 = 1) (T=1 이면 기존과 동일)
            last = (i == steps - 1)
            resample = ((p1 > 0.5).float() if last
                        else (torch.rand_like(p1) < p1).float())
            a = jumps[i]                                     # (κ_{t'}-κ_t)/(1-κ_t)
            do = torch.rand(B, num_bits, device=device) < a
            x = torch.where(do, resample, x)                 # 유지 vs 예측 쪽 점프
        return x.long()
    else:
        # --- Grouped token sampling ---
        vocab_size = 2 ** group_size
        num_tokens = num_bits // group_size
        # exact finite-step transition (state-preserving), token-level.
        x = torch.randint(0, vocab_size, (B, num_tokens), device=device)
        for i in range(steps):
            t = torch.full((B,), alphas[i], device=device)
            logits = _cfg_logits(model, x, t, y, guidance_scale)  # (B,T,V)
            if flip_pred:
                logits = _apply_flip_residual(logits, x, vocab_size, t)
            probs = F.softmax(logits / temperature, dim=-1)
            last = (i == steps - 1)
            if last:
                resample = probs.argmax(-1)
            else:
                resample = torch.multinomial(
                    probs.reshape(-1, vocab_size), 1).reshape(B, num_tokens)
            a = jumps[i]                                     # (κ_{t'}-κ_t)/(1-κ_t)
            do = torch.rand(B, num_tokens, device=device) < a
            x = torch.where(do, resample, x)                 # 유지 vs 예측 쪽 점프
        return tokens_to_bits(x, group_size).long()


def _cfg_logits(model, x_input, t, y, guidance_scale):
    """CFG logit computation (shared by binary and grouped sampling)."""
    logits_cond = model(x_input, t, y)
    if guidance_scale != 1.0:
        logits_uncond = model.forward_uncond(x_input, t)
        return logits_uncond + guidance_scale * (logits_cond - logits_uncond)
    return logits_cond


# ===========================================================================
# Absorbing (mask) DFM — MDLM/MD4 스타일
# ---------------------------------------------------------------------------
# uniform-replacement DFM 과 달리, 노이즈는 심볼을 uniform 재추출하는 대신
# 흡수(absorbing) 상태 MASK 로 보낸다. 모델은 mask_token=True 로 build 한다:
#   입력 임베딩 = V+1 (MASK id = V),  출력 헤드 = V (실토큰만 — MASK 는 예측 안 함).
# 따라서 CE·softmax 는 모두 V-way 이고, 샘플러가 MASK 를 뽑을 일이 없다.
# ===========================================================================

def _mask_corrupt_tokens(x1, t, mask_id):
    """Absorbing noising: 확률 κ_t=t 로 유지, 아니면 MASK(id=mask_id)로 보낸다."""
    B, T = x1.shape
    keep = torch.rand(B, T, device=x1.device) < t.view(B, 1)
    return torch.where(keep, x1, torch.full_like(x1, mask_id))


def dfm_mask_loss(model, y, x1, group_size=1):
    """Absorbing DFM NELBO. 모델 헤드는 V-way (MASK 제외).
    x1: (B, num_bits) in {0,1} float."""
    B = x1.size(0)
    V = 2 ** group_size                          # 실제 vocab (MASK 제외)
    MASK = V                                     # 입력 임베딩의 마지막 행 id
    tokens = bits_to_tokens(x1, group_size) if group_size > 1 else x1.long()
    t = torch.rand(B, device=x1.device)
    tokens_t = _mask_corrupt_tokens(tokens, t, MASK)

    logits = model(tokens_t, t, y)               # (B, T, V)
    masked = (tokens_t == MASK).float()          # (B, T) — masked 위치만 loss

    ce = F.cross_entropy(logits.reshape(-1, V), tokens.reshape(-1),
                         reduction="none").reshape(B, -1)
    w = 1.0 / (1.0 - t).clamp_min(1e-3)          # κ'/(1-κ) = 1/(1-t)
    loss = (ce * masked).sum() / masked.sum().clamp_min(1.0)
    return loss.mean() / tokens.size(1)


@torch.no_grad()
def sample_dfm_mask_cfg(model, y, num_bits, steps=50, guidance_scale=1.0,
                        group_size=1, schedule="linear", temperature=1.0):
    """Absorbing 샘플링: 전부 MASK 에서 시작, 매 스텝 masked 위치만 비가역적으로 해제.

    점프율/모델 t 입력은 uniform DFM 과 동일하게 `_keep_schedule` 로 스케줄과 정합.
    """
    model.eval()
    device = next(model.parameters()).device
    y = y.to(device)
    B = y.size(0)
    V = 2 ** group_size
    MASK = V
    T = num_bits // group_size if group_size > 1 else num_bits
    alphas, jumps = _keep_schedule(steps, schedule)

    x = torch.full((B, T), MASK, device=device, dtype=torch.long)
    for i in range(steps):
        t = torch.full((B,), alphas[i], device=device)
        logits = _cfg_logits(model, x, t, y, guidance_scale)      # (B, T, V) — MASK 없음
        probs = F.softmax(logits / temperature, dim=-1)
        if i == steps - 1:
            cand = probs.argmax(-1)                               # 마지막: 결정론
        else:
            cand = torch.multinomial(probs.reshape(-1, V), 1).reshape(B, T)
        a = jumps[i]
        do = (torch.rand(B, T, device=device) < a) & (x == MASK)  # masked 만 해제
        x = torch.where(do, cand, x)                              # carry-over: 비가역
    return tokens_to_bits(x, group_size).long() if group_size > 1 else x.long()
