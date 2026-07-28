# Binary Image Generative Model — 설계 변인 정리

최종 출력이 각 픽셀 {0,1}인 바이너리 이미지(antenna metal mask 등)를 생성하는
diffusion / flow matching 모델의 설계 변인, 각 선택지의 대표 문헌, 그리고
우리 구현이 채택한 방법을 정리한다.

- **본선**: Discrete Flow Matching (DFM)
- **baseline**: Continuous Flow Matching (CFM, Analog Bits식 연속 완화)

관련 코드: [`flow_matching.py`](./flow_matching.py)

---

## 결정 트리 (가지치기)

변인들은 독립이 아니라 **종속**이다. 첫 선택(상태공간)이 source·이산화·예측·loss를
대부분 자동으로 결정(prune)하므로, flat한 표보다 트리로 보면 결정이 쉽다.

```
바이너리 이미지 생성
│
[1] 상태공간?  ← 이 한 번의 선택이 source·이산화·예측·loss를 대부분 강제
│
├─ A. 연속 relax (ℝ; {0,1}→{-1,+1})
│    ├─ source ....... Gaussian     ★연속에서만 정의됨
│    ├─ 이산화 ....... 끝에서 threshold  (강제 — 중간은 실수)
│    ├─ loss ......... MSE
│    ├─ [2] 시간?
│    │     ├─ 이산 → DDPM류          (예측: ε / x0 / v)
│    │     └─ 연속 → Score-SDE / FM  (예측: score / v)   ★우리 CFM = 연속+v
│    ├─ [7] 샘플링 noise?  deterministic ODE ★우리 / stochastic SDE(ancestral·churn)
│    └─ 문헌: DDPM, Analog Bits, Rectified Flow
│
├─ B. 이산 {0,1}                                    ★★우리 본선
│    ├─ source ....... uniform  또는  mask/absorbing   (Gaussian 불가)
│    ├─ 이산화 ....... 매 스텝 이산  (자동)
│    ├─ 예측 ......... x0·x1 posterior / Bernoulli p / rate / score-ratio  (ε·v 불가)
│    ├─ loss ......... BCE / CE
│    ├─ [2] 시간?
│    │     ├─ 이산 → D3PM, BerDiff
│    │     └─ 연속 →
│    │          ├─ [3] rate 얻는 법?
│    │          │      ├─ noising 정의→역전 ⇒ CTMC diffusion  (rate 유일)
│    │          │      └─ path 처방→continuity ⇒ DFM (rate 비유일)  ★우리
│    │          ├─ [4] source?   uniform ★우리(_corrupt_bits) / mask
│    │          ├─ [5] coupling? independent ★우리 / OT(옵션·이득 작음)
│    │          ├─ [6] sampler?  tau-leaping(Euler) / exact 유한전이 ★우리
│    │          └─ [7] noise?    η=0(최소·argmax 마감) ★우리 / η>0(corrector·re-masking)
│    └─ 문헌: D3PM, BerDiff, CTMC, DFM(Gat/Campbell 2024)
│
└─ C. 확률 simplex / logit
     ├─ source ....... Dirichlet / logit-space Gaussian
     ├─ 이산화 ....... 샘플링 순간 (argmax·Bernoulli sample)
     └─ 문헌: Dirichlet Diffusion, Argmax Flows
```

### 강제 종속 관계 (왜 가지치기가 되는가)

| 상위 선택 | 자동으로 결정/배제되는 것 |
|---|---|
| **Gaussian source** | ⟺ **연속 relax에서만** (이산 상태엔 Gaussian 노이즈 정의 불가) |
| **연속 relax** | ⟹ 이산화는 **끝에서만** (중간이 실수라 "매 스텝 이산" 불가) |
| **이산 상태** | ⟹ **ε·velocity 회귀 배제** (이산 점프엔 방향벡터 없음) ⟹ posterior/rate 예측 |
| **이산 상태** | ⟹ 매 스텝 이산 가능, loss는 BCE/CE |

핵심: **[1] 상태공간만 정하면** source·이산화·예측대상·loss가 거의 따라온다.
남는 자유 변인은 [2]시간, [3]rate 방식, [6]샘플러 정도.

### 추가 변인 [7]: 샘플링 stochasticity (노이즈 주입 여부)

학습 모델·marginal이 같아도 **샘플링을 deterministic으로 할지 stochastic으로 할지**는
독립적인 설계 선택이며 품질/다양성에 크게 영향.

- **연속(A)**: 같은 marginal의 **ODE(deterministic) ↔ SDE(stochastic)** 쌍 (Song 2021).
  - deterministic: 재현 가능, few-step, invertible. ← 우리 CFM (`sample_fm`)
  - stochastic: 매 스텝 노이즈 주입 → 누적오차 자가교정, 다양성↑.
- **이산(B)**: 완전 deterministic flow는 없음(점프 과정). **stochasticity 양 η**가 손잡이.
  - η=0(최소): 생성 rate만, 마지막 argmax. ← 우리 DFM 현재
  - η>0: 이동 후 일부 bit 재노이즈→재denoise (corrector/re-masking, Campbell 2024).
    marginal 보존하며 mixing↑ → 오차교정·품질↑. CTMC-diffusion은 특정 η에 해당.
  - temperature: posterior sharpen(greedy) ↔ flatten(diverse).

**용도별 선택**: "최적 mask 하나" → deterministic/low-temp; "후보 다수 뽑아 시뮬 선별"
→ stochastic(η>0); few-step 빡셀 때 → 약간의 stochasticity가 보통 유리.

### 우리 최종 경로 (트리 상의 한 줄)

- **본선**: `B(이산) → 연속시간 → DFM → uniform source → independent coupling → exact 유한전이 sampler → η=0`
- **baseline**: `A(연속 relax) → 연속시간 → velocity 예측 → threshold → deterministic ODE`

---

## 표 1. 설계 변인 × 선택지 × 문헌 × 우리 채택

| 변인 | 선택지 | 대표 문헌 | 우리 채택 |
|---|---|---|---|
| **① 시간 연속성** | 이산 시간 | DDPM (Ho 2020), D3PM (Austin 2021), BerDiff (Chen 2023) | |
| | **연속 시간** | Score-SDE (Song 2021), Flow Matching (Lipman 2023), CTMC (Campbell 2022), DFM (Gat/Campbell 2024) | ✅ **DFM & CFM 둘 다 연속** |
| **② 상태 공간** | 연속 relax (ℝ) | DDPM, Rectified Flow (Liu 2023), **Analog Bits (Chen 2023)** | ◻︎ CFM baseline ({-1,+1}) |
| | **이산 상태 {0,1}** | D3PM, BerDiff, CTMC, **DFM** | ✅ **DFM 본선** |
| | 확률/simplex | Dirichlet Diffusion (Avdeyev 2023), Argmax Flows (Hoogeboom 2021) | |
| **③ Source(prior)** | Gaussian | DDPM, FM, Rectified Flow | ◻︎ CFM |
| | **Uniform categorical** | D3PM-uniform, Multinomial Diff (Hoogeboom 2021), **DFM-uniform** | ✅ **DFM (`_corrupt_bits`)** |
| | Absorbing / mask | D3PM-absorbing, SEDD (Lou 2024), MaskGIT (Chang 2022) | |
| | OT coupling | Rectified Flow, **DFM w/ coupling** | ◻︎ 옵션 (`_ot_corrupt_bits`) |
| **④ 예측 대상** | ε (noise) | DDPM | |
| | velocity $v$ | **Flow Matching, Rectified Flow** | ◻︎ **CFM (`cfm_loss`)** |
| | score / ratio | Score-SDE, SEDD (concrete score) | |
| | rate matrix | (직접 예측은 드묾) Campbell 2022 일부 | |
| | Bernoulli $p$ (flip) | BerDiff | |
| | **$x_0/x_1$ posterior** | D3PM, DiGress (Vignac 2023), CTMC, **DFM** | ✅ **DFM (`dfm_loss`)** |
| **⑤ 이산화 시점** | 끝에서 threshold | Analog Bits, CFM류 | ◻︎ CFM (`pm1_to_bits01`) |
| | **매 스텝 이산** | D3PM, BerDiff, CTMC, **DFM** | ✅ **DFM** |
| | 샘플링 순간 | Concrete/Gumbel, Argmax Flows | |
| **⑥ Loss** | MSE | DDPM, FM | ◻︎ CFM |
| | **BCE / CE** | D3PM (VLB+CE), DiGress, **DFM** | ✅ **DFM (BCE / grouped CE)** |
| | score matching | Score-SDE, SEDD | |
| **⑦ Sampler** | ODE Euler | FM, Rectified Flow | ◻︎ CFM (`sample_fm`) |
| | ancestral posterior | DDPM, D3PM | |
| | tau-leaping (rate) | Campbell 2022, DFM(Euler) | |
| | **exact 유한전이** | DFM 계열 (closed-form step) | ✅ **DFM (`sample_dfm_cfg`)** |
| **⑧ 샘플링 noise** | deterministic (ODE / argmax) | DDIM (Song 2021), FM Euler | ✅ **CFM ODE / DFM η=0** |
| | stochastic (SDE / η>0 corrector) | Ancestral·SDE (Song 2021), DFM corrector (Campbell 2024) | ◻︎ 미구현 (η 손잡이 예정) |

---

## 표 2. 우리 구현 = 축 조합 요약

| | **DFM (본선)** | CFM (baseline) |
|---|---|---|
| 시간 | 연속 | 연속 |
| 상태 | 이산 {0,1} / grouped token | 연속 relax {-1,+1} |
| prior | uniform (또는 OT) | Gaussian |
| 예측 | $x_1$ posterior | velocity $v$ |
| loss | BCE / CE | MSE |
| 이산화 | 매 스텝 | 끝에서 threshold |
| sampler | exact 유한전이 (state-preserving) | Euler ODE |
| 대응 문헌 | Gat 2024 / Campbell 2024 (DFM) | Lipman 2023 / Liu 2023 |

---

## CTMC vs DFM (참고)

둘 다 연속시간·이산상태·샘플링=CTMC 시뮬레이션·학습타깃=$x_1$ posterior로 **공통**.
차이는 **rate matrix를 얻는 원리**뿐이다.

| | CTMC diffusion | DFM |
|---|---|---|
| 발상 | forward noising 정의 → 시간 역전으로 rate 유도 | source→data 경로 처방 → continuity eq로 rate 유도 |
| rate | forward에 묶여 **유일** | **비유일** (stochasticity 자유도) |
| 연속 대응 | score-SDE | flow matching |
| 포함관계 | DFM의 특수 케이스 | 상위집합 |

- **우리는 DFM 채택**: uniform-source 경로(`keep-prob-t`)를 직접 처방하고, 예측
  $\hat{p}(x_1)$에서 generating rate를 유도. CTMC-diffusion(masked/uniform)은
  DFM의 특수 케이스로 포함된다.

### 샘플러: exact 유한전이

예측 $\hat{p}(x_1)$을 스텝 동안 고정하면 구간 CTMC를 닫힌형태로 적분 가능:

$$a = \frac{t' - t}{1 - t} = \frac{1}{\text{steps} - i} \quad\Rightarrow\quad
\begin{cases} \text{prob } a: & \hat{p}(x_1)\text{에서 재샘플} \\ \text{prob } 1-a: & \text{현재 } x_t \text{ 유지} \end{cases}$$

- 상태 보존(state-preserving) → 매 스텝 $x_t$를 버리지 않음.
- **Euler/tau-leaping과의 관계 (정정)**: 일반 스케줄에서 exact는
  $a = \frac{\kappa_{t'} - \kappa_t}{1-\kappa_t}$, Euler는 $a \approx \frac{\dot\kappa_t \Delta t}{1-\kappa_t}$.
  **linear $\kappa_t = t$ 에서는 두 값이 정확히 일치**하므로 현재 설정에서 exact ≡ Euler다.
  정확적분의 실익은 **비선형 $\kappa_t$**(cosine 등)에서만 발생한다.
  이번 교체의 실제 이득은 "Euler보다 정확해서"가 아니라, 이전 샘플러가
  **$x_t$를 버리고 uniform으로 재노이즈**하던 (CTMC 커널이 아닌) 방식이었기 때문이다.
- (확장) 유지분에 재노이즈 항을 얹어 stochasticity η 조절 가능 → η 특정값이 CTMC-diffusion에 해당.
- 유도 전문: [`FORMULATION.md` §7](./FORMULATION.md)

---

## 학습 vs 샘플링 분리

DFM은 학습과 샘플링이 분리된다.
- **학습** (`dfm_loss`): corruption된 $x_t$에서 clean $x_1$ posterior 예측 (BCE/CE). 샘플러와 무관.
- **샘플링** (`sample_dfm_cfg`): 학습된 posterior로 전이 구성. 재학습 없이 교체 가능.
- 따라서 샘플러 변경(→ exact 유한전이)은 **학습에 영향 없음**. 단, steps/CFG는 새 샘플러 기준으로 재스윕 권장.

---

## 가지치기와 분리된 변인 (학습·추론 하이퍼)

아래는 상태공간 트리와 **독립적**(어느 leaf에서도 자유롭게 조절 가능)이라 트리에서 분리해 정리한다.

### noise schedule κ_t

corruption keep-prob 스케줄. 현재 **linear** (keep = t). 대안: cosine, polynomial 등.
- 상태공간 선택과 무관 → 별도 변인.
- **학습과 샘플링에 동일 스케줄**을 써야 정합.
- 영향: 어느 t 구간에 모델 용량을 집중할지. discrete에서는 후반(高 t) 집중형이 유리한 경우가 있음.
- 우리: linear 채택 (미검증, ablation 대상).

### CFG (classifier-free guidance) scale

추론 전용 손잡이. `uncond + s·(cond − uncond)` (`_cfg_logits`).
- s = 1: 순수 conditional. s↑: y 충실도↑, 다양성↓.
- 전제: 학습 시 `drop_prob`로 uncond 병행 학습 (이미 구현됨, `null_y_embed`).
- ⑧ 샘플링 stochasticity와 같은 "추론 손잡이" 층 — 학습 재실행 없이 스윕.

---

## 평가 (예비)

leaf(설계 조합)를 고르는 기준. 아직 확정 아님 — 우선 아래 3개로 시작.

| 지표 | 정의 | 성격 |
|---|---|---|
| **bit accuracy** | 생성 mask vs GT mask의 per-bit 일치율 | 재현/sanity |
| **pattern accuracy** | 전체 패턴 exact match 비율 (all bits correct) | 재현/sanity (엄격) |
| **surrogate spectrum error** | \|surrogate(x_gen) − y\| — 학습된 forward 모델로 스펙트럼 재현 오차 | **진짜 목표 지표** |

**중요 caveat**: inverse design은 보통 **one-to-many** (다른 mask가 같은 스펙트럼 y를 냄).
따라서 bit/pattern acc는 유효한 대안 해를 오답으로 penalize → **재현 sanity 지표로만** 해석하고,
**surrogate로 "생성 mask가 목표 y를 실제로 내는가"**를 주 지표로 삼는다.
(surrogate = mask→spectrum forward 예측기; 전체 EM 시뮬 없이 루프를 닫아 평가.)

향후 후보: 다양성(동일 y에 대한 생성 분포), few-step 저하 곡선, 제약 유효성
(DFM은 항상 {0,1}이라 자동 충족 → 우선순위 낮음).

---

## References

- **DDPM** — Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
- **Score-SDE** — Song et al., *Score-Based Generative Modeling through SDEs*, ICLR 2021.
- **v-prediction** — Salimans & Ho, *Progressive Distillation for Fast Sampling*, ICLR 2022.
- **Flow Matching** — Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023.
- **Rectified Flow** — Liu et al., *Flow Straight and Fast*, ICLR 2023.
- **Stochastic Interpolants** — Albergo & Vanden-Eijnden, 2023.
- **Analog Bits** — Chen, Zhang, Hinton, *Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning*, ICLR 2023.
- **D3PM** — Austin et al., *Structured Denoising Diffusion Models in Discrete State-Spaces*, NeurIPS 2021.
- **Multinomial Diffusion / Argmax Flows** — Hoogeboom et al., NeurIPS 2021.
- **Dirichlet Diffusion** — Avdeyev et al., 2023.
- **BerDiff** — Chen et al., *BerDiff: Conditional Bernoulli Diffusion Model for Medical Image Segmentation*, MICCAI 2023.
- **Blackout Diffusion** — Santos et al., ICML 2023.
- **CTMC discrete diffusion** — Campbell et al., *A Continuous Time Framework for Discrete Denoising Models*, NeurIPS 2022.
- **SEDD** — Lou et al., *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*, ICML 2024.
- **Discrete Flow Matching** — Gat et al., 2024; Campbell et al., *Generative Flows on Discrete State-Spaces*, ICML 2024.
- **DiGress** — Vignac et al., *DiGress: Discrete Denoising Diffusion for Graph Generation*, ICLR 2023.
- **MaskGIT** — Chang et al., *Masked Generative Image Transformer*, CVPR 2022.
