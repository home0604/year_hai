# Flow Matching for Binary Antenna Inverse Design — 연구노트

**날짜**: 2026-07-05  
**목표**: 10x10 patch-antenna inverse design에서 DFM(Discrete Flow Matching) 학습 실패 원인 분석 및 개선

---

## 1. 문제 상황

### 이전 실험 결과 (DFM, group_size=1)

SmallDiT + DFM, Stable3DiT + DFM 두 모델 모두 학습 실패:

| Model | Epoch | Train Loss | Val Loss | BitAcc | PatAcc |
|-------|-------|-----------|----------|--------|--------|
| SmallDiT (32.9M) | 140/500 | 0.460 | 0.459 | **0.54** | 0.00 |
| Stable3DiT (76.8M) | 45/500 | 0.464 | 0.463 | **0.54** | 0.00 |

- Loss는 0.693(random) → 0.46으로 하락하지만, BitAcc 54%는 random(50%)에 가까움
- 모델 capacity 문제가 아님 (SmallDiT 32.9M vs AR Transformer 32.4M, 유사한 규모)

---

## 2. 원인 분석: Binary Vocab의 Weak Corruption

### 핵심 문제: Collision Probability

DFM corruption 과정에서 각 token을 uniform random으로 교체할 때, 교체된 값이 원래와 같을 확률:

$$P(\text{collision}) = \frac{1}{V}$$

| Vocab Size | Collision Prob | 비고 |
|-----------|---------------|------|
| V = 50,000 (언어) | 0.002% | corrupted 위치 명확히 구분 |
| V = 1,024 | 0.1% | |
| V = 16 | 6.3% | |
| **V = 2 (binary)** | **50%** | **corrupted vs clean 구분 불가** |

### 정보이론적 해석

- t=0 (최대 noise)에서도 50%의 bit가 이미 정답 → 모델이 볼 수 있는 noise range가 [0.5, 1.0]으로 압축
- 각 위치에서 복원해야 할 정보량: log₂V bits
  - Binary: 1 bit/position → timestep당 gradient signal 극도로 미약
  - V=1024: 10 bits/position → timestep당 signal 풍부
- **총 정보량은 100 bits로 동일**하지만, position당 정보 밀도가 다름

---

## 3. 해결 방안 1: Bit Grouping

### 아이디어

연속된 bit들을 묶어서 하나의 token으로 처리 → effective vocab size 증가.

| group_size | Tokens | Vocab (V) | Collision | Model Params |
|-----------|--------|-----------|-----------|-------------|
| 1 (현재) | 100 | 2 | 50% | 32.92M |
| 4 | 25 | 16 | 6.3% | 32.89M |
| 5 | 20 | 32 | 3.1% | 32.91M |
| 10 | 10 | 1024 | 0.1% | 33.92M |

### 구현

- `bits_to_tokens(x_bits, group_size)`: (B, 100) {0,1} → (B, T) {0,...,2^gs - 1}
- `tokens_to_bits(tokens, group_size)`: 역변환
- **DFM**: token-level corruption + cross-entropy loss over V classes
- **CFM**: group_size개의 continuous {-1,+1} 값을 하나의 position으로 → Linear(gs, d_model) 입력
- Model 변경:
  - DFM grouped: `Embedding(V, d_model)` 입력, `Linear(d_model, V)` 출력
  - CFM grouped: `Linear(gs, d_model)` 입력, `Linear(d_model, gs)` 출력 (CoBit의 "analog bits"와 유사)

---

## 4. 해결 방안 2: Optimal Transport Coupling

### 아이디어

기존 independent coupling에서는 x₀가 완전 random → t가 낮을 때 x_t가 uninformative.  
Mini-batch OT로 x₀를 x₁에 가까운 패턴과 매칭하면 transport path가 짧아짐.

### 구현

- Hamming distance 기반 cost matrix 계산 (GPU)
- `scipy.optimize.linear_sum_assignment` (Hungarian algorithm) — exact OT
- Overhead: ~1ms/batch (B=128), epoch당 ~4.5초 (무시 가능)

### 효과 측정 (B=128, D=100)

| Coupling | Avg Hamming from x₁ | Reduction |
|----------|---------------------|-----------|
| Independent | 25.1 | - |
| OT (exact) | 19.0 | **24.3%** |

---

## 5. Literature Survey

### Binary Discrete Diffusion 관련 논문

Binary vocab(V=2)에서의 discrete diffusion을 정면으로 다룬 논문은 극소수. 주요 발견:

#### CoBit (Batzolis et al., 2026) — arXiv 2605.07013
- **"Towards Closing the Autoregressive Gap in Language Modeling via Entropy-Gated Continuous Bitstream Diffusion"**
- m = ceil(log₂V) bits를 하나의 token으로 grouping ("semantic bit-patching")
- **Naive bit-level discrete diffusion이 catastrophically 실패함을 실증** (GenPPL 285 vs token-level 126)
- Continuous diffusion on analog bits (grouped) → GenPPL 59.8
- **우리의 grouping 접근을 직접 검증하는 논문**

#### 6가지 해결 전략 (문헌 종합)

| 전략 | 핵심 아이디어 | 대표 논문 |
|------|-------------|----------|
| Absorbing state (MASK) | 3번째 state 추가 → corruption 명확 | D3PM, MDLM |
| Flip prediction | bit 변경 여부를 예측 (residual learning) | Pham et al. (ICML 2025) |
| Analog Bits | binary → continuous → Gaussian diffusion → threshold | Chen & Lipman (2023) |
| Bit grouping | bits를 token으로 묶어 vocab 증가 | CoBit |
| 비대칭 noise schedule | 0→1, 1→0 flip rate를 다르게 | BDPM |
| Cosine schedule | informative 구간에 step 집중 | - |

#### 기타 관련 논문

- **D3PM** (Austin et al., NeurIPS 2021): Discrete diffusion 일반 framework, Bernoulli diffusion 포함
- **UDLM** (Schiff et al., ICLR 2025): small vocab에서 uniform noise가 absorbing보다 같거나 나음 (self-correction 가능)
- **DIFUSCO** (NeurIPS 2023), **DiffUCO** (ICML 2024): binary combinatorial optimization에 Bernoulli diffusion 적용
- **안테나/topology optimization에 discrete diffusion 적용한 논문은 없음** — open gap

### Flip Prediction vs Posterior 차이

| | Posterior (현재 방식) | Flip Prediction |
|--|---------------------|----------------|
| 예측 대상 | P(x₁ = 1 \| x_t) | P(x_t ≠ x₁ \| x_t) |
| Target | x₁ (원본 데이터) | (x_t ≠ x₁).float() |
| 특징 | 정답 자체를 맞추기 | 현재 상태에서 잘못된 bit 찾기 |
| Binary에서 이점 | 약한 gradient (입력 복사 학습) | residual learning으로 명확한 signal |

수학적으로 동치: P(x₁=1) = (1-f)·x_t + f·(1-x_t), 단 학습 dynamics가 다름.

---

## 6. 현재 진행 중인 실험 (2026-07-05)

wandb project: `rogers_inverse_dit`

| GPU | Architecture | Flow | group_size | V | Run Name |
|-----|------------|------|-----------|---|----------|
| 4 | SmallDiT | DFM | 4 | 16 | `fm10x10-smalldit-dfm-g4-...` |
| 5 | SmallDiT | DFM | 10 | 1024 | `fm10x10-smalldit-dfm-g10-...` |
| 6 | Stable3DiT | DFM | 4 | 16 | `fm10x10-stable3dit-dfm-g4-...` |
| 7 | SmallDiT | CFM | 4 | 16 | `fm10x10-smalldit-cfm-g4-...` |

### 비교 목적

- **GPU 4 vs 5**: group_size 4 vs 10 비교 (sequence length / vocab tradeoff)
- **GPU 4 vs 6**: SmallDiT vs Stable3DiT 비교 (architecture capacity)
- **GPU 4 vs 7**: DFM vs CFM 비교 (같은 grouping, discrete vs continuous)

### 공통 설정

- d_model=512, L=15, nhead=4, ff=768
- lr=1e-4, Adam, warmup 5ep + cosine decay
- batch_size=128, epochs=500, drop_prob=0.1
- Data: `/hai/home/lsh/antenna/year_hai/data/datasets/`

---

## 7. Stable3DiT Architecture (2026-07-07)

### 모델 구성

```
Stable3DiT (SD3/MMDiT-style)
├── Input: Embedding(V=16, 512)     ← token index → d_model
├── PosEmbed: Embedding(25, 512)    ← 25 token positions
├── t_mlp: sinusoidal(512) → SiLU → Linear(512,512)
├── y_encoder: ResNet1D(3 blocks) → AdaptivePool(50) → (B,50,512)
│   └── 201-point spectrum → 50-token sequence for cross-attn KV
├── Stable3DiTBlock × 15
│   ├── adaLN modulation: SiLU → Linear(512, 3584)  → γ₁β₁α₁ α₂ γ₃β₃α₃
│   ├── Self-Attention:    LN·modulate → MHA(4 heads, d_head=128) → α₁ gate
│   ├── Cross-Attention:   LN(Q=x), LN(KV=y_seq) → MHA → α₂ gate
│   └── FFN:               LN·modulate → Linear(768) → SiLU → Linear(512) → α₃ gate
├── Final adaLN + LayerNorm
└── Output: Linear(512, 16)          ← logits over V=16 vocab
```

### 주요 하이퍼파라미터

| 항목 | 값 | 비고 |
|------|-----|------|
| d_model | 512 | |
| num_layers | 15 | |
| nhead | 4 | d_head = 128 |
| dim_feedforward | 768 | **1.5× d_model** (일반적으론 4×) |
| vocab_size | 16 | 2⁴ (group_size=4) |
| seq_len (x) | 25 | 100 bits / 4 |
| seq_len (y) | 50 | cross-attn KV 길이 |
| drop_prob | 0.1 | CFG unconditional training |

### 파라미터 분해 (76.81M total)

| Module | Params | 비율 |
|--------|--------|------|
| blocks × 15 | 70,974,720 | 92.4% |
| ├ adaLN modulation (×15) | 27,578,880 | **35.9%** |
| ├ self_attn (×15) | 15,759,360 | 20.5% |
| ├ cross_attn (×15) | 15,759,360 | 20.5% |
| └ ffn (×15) | 11,815,680 | 15.4% |
| y_encoder (ResNet1D) | 4,730,880 | 6.2% |
| t_mlp | 525,312 | 0.7% |
| final_adaLN | 525,312 | 0.7% |
| input/pos embed | 20,992 | <0.1% |
| output_proj | 8,208 | <0.1% |

### adaLN modulation이 35.9%로 높은 이유

일반적인 DiT에서 adaLN은 ~30% 차지. 우리가 더 높은 이유:

1. **Modulation 수 = 7** (표준 DiT는 6). Cross-attn gate α₂가 추가됨
2. **FFN dim이 작다** — 768 = 1.5× d_model. 표준은 4× = 2048

d_model² 단위 비교:

| 모듈 | 우리 (ff=1.5×) | 표준 (ff=4×) |
|------|-------------|------------|
| Self-attn | 4d² | 4d² |
| Cross-attn | 4d² | 4d² |
| FFN | **3d²** | **8d²** |
| adaLN (7개) | 7d² | 7d² |
| adaLN 비율 | **7/18 = 39%** | 7/23 = 30% |

FFN이 좁으니까 adaLN 비중이 상대적으로 커진 것. AR baseline(ff=768)과 맞추려고 한 설정이지만, DiT 관점에서는 FFN이 underpowered할 수 있음.

### AR Transformer Baseline과 비교

| | AR Transformer | Stable3DiT |
|--|---------------|-----------|
| d_model | 512 | 512 |
| Layers | 15 | 15 |
| FFN dim | 768 | 768 |
| Heads | 4 | 4 |
| Attention | causal self-attn | self-attn + cross-attn |
| Conditioning | additive (y_vec) | adaLN-Zero(t) + cross-attn(y_seq) |
| Generation | autoregressive (100 steps) | iterative denoising (**1-shot OK**) |
| Total params | ~32.4M | **76.8M (2.4×)** |
| 차이 원인 | — | cross-attn(15.8M) + adaLN(27.6M) + SeqEncoder(4.7M) |

---

## 8. Sampling 방법론 비교 (2026-07-07)

### 4가지 이산 데이터 생성 방법

#### Method 1: CFM (Continuous Flow Matching)
- {0,1} → {-1,+1} continuous relaxation
- x_t = (1-t)·x₀ + t·x₁, x₀ ~ N(0,1)
- Velocity prediction + MSE loss
- Euler ODE → threshold at 0

#### Method 2: 기존 DFM 구현 (Re-corruption sampler)
- Forward: keep x₁ with prob t, uniform random with prob (1-t)
- Model predicts p(x₁|x_t,t) → CE loss
- **Sampling: predict → sample → 전체 state를 t_next로 re-corrupt**
  ```
  pred = sample_from(p_θ)
  rand = uniform_random()
  x_{t+h} = where(rand < t_next, pred, rand)   ← 전체 re-corruption
  ```

#### Method 3: 정식 DFM (Euler on CTMC rate) ★현재 사용
- Forward/Model/Loss: Method 2와 **동일**
- **Sampling: 각 token이 독립적으로 stay/jump 결정**
  ```
  jump_prob = h / (1 - t)                        ← (t'-t)/(1-t)
  jump_mask = rand() < jump_prob
  pred = sample_from(p_θ)
  x_{t+h} = where(jump_mask, pred, x_t)          ← 대부분 유지, 소수만 jump
  ```

#### Method 4: D3PM / Binary Latent Diffusion
- D3PM: discrete diffusion + ELBO loss + Bayes rule sampling
- Binary Latent Diffusion: binary autoencoder → latent에서 diffusion
- Bit Diffusion: Method 1과 유사 (continuous relaxation)

### Method 2 vs 3: 핵심 차이

**학습은 완전히 동일. 차이는 sampling procedure만.**

| | Re-corruption (#2) | Euler CTMC (#3) |
|--|-------------------|-----------------|
| 업데이트 방식 | 전체 state re-corrupt | 소수 token만 jump |
| x_{t+h}이 x_t에 의존? | **아니오** | **예** (대부분 유지) |
| step당 변경 비율 | 100% | h/(1-t) (~수%) |
| 이미 맞는 token | 다시 랜덤화 (낭비) | 유지됨 |
| 적은 step 성능 | 열화 큼 | **거의 영향 없음** |

### Eval 결과: Euler sampler step & CFG sweep

Checkpoint: `stable3dit-dfm-spg4` (76.81M), test set, seed=0

| steps | cfg | BitAcc | PatAcc | Time |
|-------|-----|--------|--------|------|
| 50 | 1.0 | 0.8184 | 0.1090 | 190.8s |
| 50 | 3.0 | 0.8498 | 0.2777 | 327.4s |
| 10 | 3.0 | 0.8502 | 0.2789 | 65.8s |
| 4 | 5.0 | 0.8520 | 0.2947 | 26.2s |
| 2 | 5.0 | 0.8523 | 0.2967 | 13.2s |
| **1** | **1.0** | **0.8577** | **0.3219** | **3.9s** |
| 1 | 3.0 | 0.8537 | 0.3064 | 6.6s |

**발견:**
- Step=1, cfg=1.0이 overall best (PatAcc 0.3219)
- Euler sampler에서 step 수가 거의 무의미 — 1 step으로 충분
- Step=1은 one-shot denoising (jump_prob=1.0으로 전체 jump)
- Step>1에서는 CFG가 도움 (PatAcc 0.11→0.28), step=1에서는 CFG가 오히려 해로움
- CFG>1은 ~1.7× 느림 (unconditional forward pass 추가)

---

## 9. 다음 단계 (TODO)

- [x] ~~실험 결과 확인 후 best group_size 결정~~ → group_size=4 spatial
- [x] ~~Flip prediction reparameterization 구현 및 비교~~ → 학습 중 (GPU 3)
- [x] ~~OT coupling + grouping 조합 실험~~ → OT는 CE loss에서 무의미, 폐기
- [x] ~~CFG guidance_scale > 1.0 sweep~~ → step>1에서만 유효, step=1에서는 해로움
- [x] ~~Sampling 방법론 비교~~ → Euler CTMC sampler 적용, step=1 best
- [ ] Euler sampler로 flip_pred checkpoint 평가
- [ ] group_size=1 baseline 평가 (학습 중, GPU 5)
- [ ] AR Transformer baseline과 최종 비교
- [ ] FFN dim 키우기 (768→2048) 실험 검토
- [ ] 학습 개선: gradient clipping, LR warmup fix, time importance sampling
