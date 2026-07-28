# Discrete Flow Matching for Binary Antenna Inverse Design — 수학적 정식화

현재 구현(`flow_matching.py`, `model.py`)에 대응하는 정식 정의.
설계 선택지 전체는 [`DESIGN_CHOICES.md`](./DESIGN_CHOICES.md), 실험 기록은 [`research_notes.md`](./research_notes.md).

---

## 0. 문제 설정 (Problem setup)

- **Mask (생성 대상)**: $b \in \{0,1\}^D$, $D = H\cdot W = 100$. $b_j = 1$ ⇔ 위치 $j$에 metal.
- **Condition (입력)**: $y \in \mathbb{R}^{P}$, $P = 201$ — 목표 스펙트럼 (S11 등).
- **Forward (물리)**: $\mathcal{F}: \{0,1\}^D \to \mathbb{R}^P$, 시뮬레이터 또는 surrogate.
- **Data distribution**: $q(b, y)$, 실제로는 $y = \mathcal{F}(b)$ 로 생성된 pair.
- **목표**: conditional generative model $p_\theta(b \mid y) \approx q(b\mid y)$.

> **Inverse design은 one-to-many**: $\mathcal{F}^{-1}(y)$ 는 일반적으로 다수의 유효해를 가진다.
> 따라서 ground-truth 재현($b = b^{\text{GT}}$)이 아니라 $\mathcal{F}(\hat b) \approx y$ 가 본질적 목표.
> (→ 평가 지표 설계의 근거, §10)

---

## 1. 상태공간과 어휘 (State space & vocabulary)

Binary mask를 그대로 쓰지 않고 **bit grouping**을 통해 토큰화한다.

### 1.1 공간 재배열 (permutation)

$\pi : \{1,\dots,D\} \to \{1,\dots,D\}$ — raster order를 $p_h \times p_w$ patch order로 재배열
(`make_spatial_patch_indices`). 목적: 하나의 토큰이 **공간적으로 인접한** bit들을 묶도록.

### 1.2 그룹화 (tokenization)

group size $g \mid D$ 에 대해

$$
T = \frac{D}{g} \quad(\text{토큰 수}), \qquad
V = 2^{g} \quad(\text{어휘 크기}), \qquad
\mathcal{S} = \{0, 1, \dots, V-1\}.
$$

$$
\phi_g : \{0,1\}^D \to \mathcal{S}^T, \qquad
\big[\phi_g(b)\big]^{(k)} \;=\; \sum_{j=0}^{g-1} 2^{\,j}\, b_{\pi(gk+j)}
\quad (k = 0,\dots,T-1)
$$

little-endian binary→integer (`bits_to_tokens`). $\phi_g$ 는 **전단사(bijection)**, 역사상 $\psi_g = \phi_g^{-1}$ (`tokens_to_bits`).

### 1.3 상태공간

$$
\boxed{\;\mathcal{X} \;=\; \mathcal{S}^{T} \;=\; \{0,\dots,V-1\}^{D/g}, \qquad |\mathcal{X}| = V^{T} = 2^{D}\;}
$$

**중요**: $|\mathcal{X}| = 2^D$ 는 $g$ 와 **무관**하다. 그룹화는 상태공간의 *크기*를 바꾸는 것이 아니라
**인수분해(factorization) 방식**을 바꾼다 — 즉 모델이 어떤 상관관계를 *명시적으로* 표현하고
어떤 것을 *조건부 독립 가정*으로 버리는지를 바꾼다. (→ §4.2, §3)

| $g$ | $V$ | $T$ | 비고 |
|---|---|---|---|
| 1 | 2 | 100 | naive bit-level (실패, BitAcc≈0.54) |
| 4 | 16 | 25 | $2\times2$ patch |
| 10 | 1024 | 10 | |

표기: 상태 $x \in \mathcal{X}$, 토큰 $x^{(k)} \in \mathcal{S}$. 데이터 $x_1 = \phi_g(b)$.

---

## 2. 확률경로와 커플링 (Probability path & coupling)

### 2.1 Source (prior)

$$
p_0 \;=\; \mathrm{Unif}(\mathcal{S})^{\otimes T}, \qquad p_0(x) = V^{-T} = 2^{-D}.
$$

(absorbing/mask source가 아니라 **uniform** source. 코드: `torch.randint(0, V, ...)`)

### 2.2 Conditional path (mixture path)

스케줄 $\kappa: [0,1] \to [0,1]$, $\kappa_0 = 0$, $\kappa_1 = 1$, $\dot\kappa_t > 0$. 현재 구현은 **linear**: $\kappa_t = t$.

각 토큰에 대해 독립적으로

$$
\boxed{\;
p_t\big(x^{(k)} \mid x_1^{(k)}\big)
\;=\; \kappa_t \,\delta\!\left(x^{(k)}, x_1^{(k)}\right) \;+\; (1-\kappa_t)\,\frac{1}{V}
\;}
$$

즉 **확률 $\kappa_t$ 로 데이터 심볼 유지, $1-\kappa_t$ 로 $\mathcal{S}$ 에서 uniform 재추출**
(`_corrupt_bits`, `_corrupt_tokens`의 `keep_mask = rand < t`).

전체 상태에 대해 조건부 독립:

$$
p_t(x \mid x_1) \;=\; \prod_{k=0}^{T-1} p_t\big(x^{(k)} \mid x_1^{(k)}\big).
$$

경계 확인: $p_0(\cdot\mid x_1) = \mathrm{Unif}$, $p_1(\cdot \mid x_1) = \delta_{x_1}$. ✔

### 2.3 Marginal path

$$
p_t(x \mid y) \;=\; \sum_{x_1 \in \mathcal{X}} q(x_1 \mid y)\, p_t(x \mid x_1).
$$

이것이 우리가 $t: 0 \to 1$ 로 시뮬레이션하려는 경로. (마진은 **인수분해되지 않음** — §4.2)

### 2.4 커플링 (coupling)

경로를 $x_0$ 를 명시해 다시 쓰면

$$
x_t^{(k)} \;=\;
\begin{cases}
x_1^{(k)} & \text{w.p. } \kappa_t\\[2pt]
x_0^{(k)} & \text{w.p. } 1-\kappa_t
\end{cases}
\qquad (x_0, x_1) \sim \Pi.
$$

- **Independent coupling** (기본): $\Pi(x_0,x_1) = p_0(x_0)\, q(x_1\mid y)$.
- **Mini-batch OT coupling** (ablation, `_ot_corrupt_bits`): 배치 $\{x_0^i\}, \{x_1^i\}$ 에 대해
  $$
  \sigma^\star = \arg\min_{\sigma \in \mathfrak{S}_B} \sum_{i=1}^{B} d_H\!\left(x_0^{\sigma(i)},\, x_1^{i}\right)
  $$
  ($d_H$: Hamming, Hungarian $O(B^3)$). 마진 $p_0, q$ 는 보존되므로 **경로 $p_t$ 는 불변**,
  분산만 감소. 측정: 평균 Hamming $25.1 \to 19.0$ ($-24\%$).

---

## 3. Corruption channel — 어휘 크기 $V$ 의 역할

토큰 단위로 보면 §2.2는 **$V$-ary symmetric channel**이다:

$$
W_t\big(v \mid u\big) \;=\; \kappa_t\,\delta(v,u) + \frac{1-\kappa_t}{V},
\qquad u = x_1^{(k)},\; v = x_t^{(k)}.
$$

### 3.1 Collision (충돌)

"재추출"이 원래 심볼을 다시 뽑을 확률 $= 1/V$. 따라서 **실제로 값이 바뀔 확률**은

$$
\Pr\!\left[x_t^{(k)} \neq x_1^{(k)}\right] \;=\; (1-\kappa_t)\,\frac{V-1}{V}.
$$

- $V=2$: 최대 $0.5$ (at $t=0$) — 즉 $x_0$ 는 **이미 절반이 정답**이지만, *어느 절반인지* 알 수 없다.
- $V=1024$: 최대 $\approx 0.999$.

### 3.2 보존 정보량 (retained information)

$x_1^{(k)}$ 가 uniform이라 가정하면 채널 용량(= 상호정보량)은

$$
I_g(t) \;=\; \log_2 V \;-\; H\big(W_t(\cdot\mid u)\big)
\;=\; g \;+\; \alpha\log_2\alpha + (V-1)\beta \log_2 \beta,
$$
$$
\alpha = \kappa_t + \tfrac{1-\kappa_t}{V}, \qquad \beta = \tfrac{1-\kappa_t}{V}.
$$

**전체** 보존 정보량 $\;\mathcal{I}_g(t) = T \cdot I_g(t) = \frac{D}{g} I_g(t)$ (bits), $D=100$:

| $t$ | $g{=}1\ (V{=}2)$ | $g{=}4\ (V{=}16)$ | $g{=}10\ (V{=}1024)$ |
|---|---|---|---|
| 0.0 | 0.0 | 0.0 | 0.0 |
| 0.2 | 2.9 | 6.5 | 12.9 |
| 0.5 | 18.9 | 29.3 | 40.1 |
| 0.8 | 53.1 | 64.3 | 72.8 |
| 1.0 | 100 | 100 | 100 |

⇒ **동일한 $\kappa_t$ 하에서 $g$ 가 클수록 $x_t$ 가 $x_1$ 에 대해 더 많은 정보를 보존**한다.
$V=2$ 의 collision($1/2$)이 corruption event의 절반을 "낭비"하면서도,
낭비된 절반이 *어느 것인지* 식별 불가능하게 만들기 때문.

> **주의(정직한 서술)**: 이 표는 $x_1$ 토큰이 i.i.d. uniform이라는 가정 하의 값이며 상한에 해당한다.
> 실제 mask는 강한 구조를 가지므로 절대값은 다르지만, $g$ 에 대한 **단조 증가 경향**은 유지된다.
> 이것이 $g=1$ 실패의 *유일한* 설명이라고 단정하지는 않는다 (§4.2의 factorization error가 병존).

---

## 4. 모델 — $x_1$-posterior (denoiser)

### 4.1 정의

신경망 $f_\theta$ 는 $(x_t, t, y) \mapsto$ logits $\ell \in \mathbb{R}^{T \times V}$ 를 내고,

$$
\boxed{\;
p_\theta\big(x_1^{(k)} = v \;\big|\; x_t, t, y\big) \;=\; \mathrm{softmax}\big(\ell^{(k)}\big)_v
\;}
$$

즉 모델은 **velocity/rate를 직접 예측하지 않고, clean data의 사후분포(posterior)를 예측**한다.
Rate는 여기서 해석적으로 유도된다 (§6).

- $V=2$ 특수화: 단일 logit $\ell^{(k)} \in \mathbb{R}$, $p_\theta(x_1^{(k)}=1\mid\cdot) = \sigma(\ell^{(k)})$.
- 입력 인코딩:
  - $g=1$: $x_t \in \{0,1\}^D \mapsto 2x_t - 1 \in \{-1,+1\}^D$ (linear proj)
  - $g>1$: token embedding $\mathcal{S}^T \to \mathbb{R}^{T\times d}$
  - $t$: sinusoidal embedding → adaLN-Zero (Stable3DiT) 또는 additive (SmallDiT)
  - $y$: 1D ResNet encoder → cross-attention (Stable3DiT) 또는 global-pooled additive (SmallDiT)

### 4.2 Factorization — 여기가 핵심 근사

모델이 표현하는 posterior는 **토큰 간 조건부 독립**:

$$
p_\theta(x_1 \mid x_t, t, y) \;=\; \prod_{k=0}^{T-1} p_\theta\big(x_1^{(k)} \mid x_t, t, y\big).
$$

그러나 **참 posterior $q(x_1 \mid x_t, y)$ 는 인수분해되지 않는다** (mask는 강하게 상관됨).

- $g=1$: 100개 bit 전부의 상관을 버림 → 근사 오차 최대.
- $g=10$: 각 토큰 **내부**의 $2^{10}$-way 상관은 **정확히** 표현. 토큰 **간** 상관만 근사.

이 근사 오차를 보상하는 유일한 장치가 **다단계 샘플링**이다: 각 스텝에서 $x_t$ 를 갱신해
다시 조건화하므로, 스텝을 나눌수록 joint를 점진적으로 복원한다.
(⇒ one-shot 생성이 실패하고 $N$-step이 필요한 이유이며, $g$ 를 키우면 필요한 보상량이 줄어든다.)

---

## 5. 학습 목적함수 (Training objective)

$$
\boxed{\;
\mathcal{L}(\theta) \;=\;
\mathbb{E}_{\,t \sim \mathcal{U}[0,1]}\;
\mathbb{E}_{\,(b,y)\sim q}\;
\mathbb{E}_{\,x_t \sim p_t(\cdot \mid x_1 = \phi_g(b))}
\left[\; -\frac{1}{T}\sum_{k=0}^{T-1} \log p_\theta\!\left(x_1^{(k)} \,\middle|\, x_t, t, y\right) \right]
\;}
$$

- $g=1$ → binary cross-entropy (`binary_cross_entropy_with_logits`)
- $g>1$ → categorical cross-entropy over $V$ (`cross_entropy`)
- $t$-weighting $w(t) \equiv 1$ (uniform). 다른 weighting은 미탐색.
- 최소해: $p_{\theta^\star}(x_1^{(k)}\mid x_t,t,y) = q(x_1^{(k)} \mid x_t, y)$ — **참 주변 posterior**.

**CFG (classifier-free guidance)**: 학습 시 확률 $p_{\text{drop}}$ 로 $y \leftarrow \varnothing$
(학습 가능한 null embedding). ⇒ 하나의 네트워크가 $p_\theta(\cdot\mid y)$ 와 $p_\theta(\cdot\mid\varnothing)$ 를 모두 학습.

---

## 6. 생성자(generator) / rate matrix

### 6.1 연속시간 마르코프 연쇄 (CTMC)

$\mathcal{X}$ 위의 CTMC를 rate matrix (generator) $u_t \in \mathbb{R}^{|\mathcal{X}|\times|\mathcal{X}|}$ 로 기술:

$$
u_t(v, z) \ge 0 \;\;(v \neq z), \qquad \sum_{v} u_t(v,z) = 0,
$$
$$
\Pr[X_{t+h} = v \mid X_t = z] = \delta(v,z) + h\, u_t(v,z) + o(h).
$$

**Kolmogorov forward equation (continuity equation)**:
$$
\frac{d}{dt} p_t(v) \;=\; \sum_{z} u_t(v,z)\, p_t(z).
$$

### 6.2 Conditional rate 유도 (토큰 단위)

$p_t(v \mid x_1) = \kappa_t \delta(v, x_1) + (1-\kappa_t)/V$ 를 만족하는 rate를 찾는다.
Ansatz $u_t(v,z\mid x_1) = \lambda_t\big[\delta(v,x_1) - \delta(v,z)\big]$ 를 대입:

$$
\sum_z u_t(v,z\mid x_1) p_t(z\mid x_1)
= \lambda_t\big[\delta(v,x_1) - p_t(v\mid x_1)\big]
= \lambda_t (1-\kappa_t)\left[\delta(v,x_1) - \tfrac{1}{V}\right].
$$

한편 $\dfrac{d}{dt}p_t(v\mid x_1) = \dot\kappa_t\left[\delta(v,x_1) - \tfrac1V\right]$. 따라서

$$
\boxed{\;
\lambda_t \;=\; \frac{\dot\kappa_t}{1 - \kappa_t}
\;\overset{\kappa_t = t}{=}\; \frac{1}{1-t},
\qquad
u_t(v, z \mid x_1) \;=\; \lambda_t\big[\delta(v,x_1) - \delta(v,z)\big].
\;}
$$

해석: **현재 상태 $z$ 를 떠나 오직 $x_1$ 로만, rate $\lambda_t$ 로 점프**한다. $t\to1$ 에서 $\lambda_t \to \infty$
(남은 토큰을 반드시 확정시킴).

### 6.3 Marginal rate

$$
\boxed{\;
u_t(v, z \mid y) \;=\; \mathbb{E}_{x_1 \sim q(\cdot \mid x_t = z,\, y)}\big[u_t(v,z\mid x_1)\big]
\;=\; \lambda_t\Big[\, \underbrace{q\big(x_1^{(k)} = v \mid x_t = z, y\big)}_{\text{모델이 예측}} - \delta(v,z)\Big]
\;}
$$

⇒ **posterior만 알면 rate는 공짜로 얻어진다.** 이것이 §4에서 posterior를 예측 대상으로 잡은 이유.
실제로는 $q \to p_\theta$ 로 대체: $u_t^\theta$.

### 6.4 DFM vs. CTMC-diffusion

| | CTMC diffusion (D3PM 계열) | DFM (본 연구) |
|---|---|---|
| 출발점 | forward noising process 정의 | probability path $p_t$ 를 **직접 규정** |
| rate 획득 | forward의 **시간 역전** | **continuity equation의 해** |
| rate 유일성 | 유일 | **비유일** (§6.5) |
| 학습 대상 | (대개) $x_0$-posterior | $x_1$-posterior |

⇒ CTMC diffusion $\subset$ DFM. 학습은 둘 다 posterior 회귀로 귀결되지만,
DFM은 **경로와 샘플러를 분리**하므로 재학습 없이 샘플러를 교체할 수 있다.

### 6.5 Rate의 비유일성 = stochasticity 손잡이

$p_t$ 를 보존하는 임의의 divergence-free 항 $w_t$ ($\sum_z w_t(v,z)p_t(z) = 0$, rate 조건 유지)에 대해
$u_t + \eta\, w_t$ 도 같은 마진을 만든다. 표준 선택은 detailed-balance 형태의 corrector
(재-노이즈 후 재-예측). **현재 $\eta = 0$** (순수 forward rate). 미구현/미탐색.

---

## 7. 샘플러 — 유한 스텝 전이 커널

### 7.1 유도

시간 격자 $0 = t_0 < t_1 < \dots < t_N = 1$. 구간 $[t, t']$ 동안 예측 posterior
$\hat p := p_\theta(x_1^{(k)} = \cdot \mid x_t, t, y) \in \Delta^{V-1}$ 를 **고정**하면,
토큰별 generator는 행렬로

$$
U_s \;=\; \lambda_s\,(\hat P - I), \qquad \hat P := \hat p\,\mathbf{1}^\top \;(\text{rank-1, 모든 열} = \hat p).
$$

$M := \hat P - I$ 는 $M^2 = \hat P^2 - 2\hat P + I = -M$ 을 만족 ($\hat P^2 = \hat P$),
따라서 $M^n = (-1)^{n-1} M$. 또한 서로 다른 $s$ 의 $U_s$ 는 교환하므로 전이 커널은

$$
P_{t\to t'} = \exp\!\left(\Lambda\, M\right), \qquad
\Lambda := \int_t^{t'} \lambda_s\, ds = \log\frac{1-\kappa_t}{1-\kappa_{t'}}.
$$

$$
\exp(\Lambda M) = I + M\sum_{n\ge1} \frac{(-1)^{n-1}\Lambda^n}{n!} = I + M\left(1 - e^{-\Lambda}\right).
$$

$1 - e^{-\Lambda} = 1 - \dfrac{1-\kappa_{t'}}{1-\kappa_t} = \dfrac{\kappa_{t'} - \kappa_t}{1-\kappa_t} \;=:\; a$. 최종:

$$
\boxed{\;
P_{t \to t'} \;=\; (1-a)\, I \;+\; a\, \hat P,
\qquad
a \;=\; \frac{\kappa_{t'} - \kappa_t}{1 - \kappa_t} \;\in [0,1]
\;}
$$

### 7.2 알고리즘

각 토큰 $k$ 에 대해 **독립적으로**:

$$
x_{t'}^{(k)} =
\begin{cases}
v \sim \hat p^{(k)} & \text{w.p. } a \quad(\text{예측 posterior에서 재추출})\\[2pt]
x_{t}^{(k)} & \text{w.p. } 1-a \quad(\textbf{현재 상태 유지})
\end{cases}
$$

균등 격자 $t_i = i/N$, linear $\kappa_t = t$ 에서
$$
a_i = \frac{1/N}{1 - i/N} = \frac{1}{N - i} \qquad (a_0 = \tfrac1N,\;\; a_{N-1} = 1).
$$

- $a_{N-1} = 1$ ⇒ **마지막 스텝은 모든 토큰을 반드시 확정** ⇒ 남은 노이즈 없음 (별도 thresholding 불필요).
- 구현(`sample_dfm_cfg`)에서 마지막 스텝은 sampling 대신 $\arg\max \hat p$ (MAP projection) — 의도적 결정론화.
- CFG: $\ell = \ell_\varnothing + s\,(\ell_y - \ell_\varnothing)$ 를 softmax 전에 적용 (`_cfg_logits`).

### 7.3 Euler / $\tau$-leaping과의 관계 — ⚠️ 정정 사항

1차 Euler(= $\tau$-leaping)는 $P_{t\to t'} \approx I + \Delta t\, U_t = (1 - \lambda_t\Delta t) I + \lambda_t \Delta t\,\hat P$,
즉 점프 확률 $a^{\text{Euler}} = \lambda_t \Delta t = \dfrac{\dot\kappa_t \,\Delta t}{1-\kappa_t}$.

$$
a^{\text{exact}} = \frac{\kappa_{t'} - \kappa_t}{1-\kappa_t},
\qquad
a^{\text{Euler}} = \frac{\dot\kappa_t\,\Delta t}{1-\kappa_t}.
$$

**linear schedule $\kappa_t = t$ 에서는 $\kappa_{t'} - \kappa_t = \Delta t = \dot\kappa_t \Delta t$ 이므로 두 값이 정확히 일치한다.**

⇒ 현재 설정(linear $\kappa$)에서 "exact 유한전이" 커널은 Euler/$\tau$-leaping과 **수치적으로 동일**하다.
정확적분이 실질적 이득을 주는 것은 **비선형 $\kappa_t$** (cosine, polynomial 등)를 쓸 때뿐이다
($\lambda_t\Delta t > 1$ 이 되어 clipping이 필요해지는 상황도 방지).

**그렇다면 이번 샘플러 교체의 실제 이득은?**
이전 구현은 Euler도 exact도 아니었다 — 매 스텝 $x_t$ 를 **버리고** uniform noise로 재구성했다:
```python
x = torch.where(rand < t_next, pred, rand)   # 이전 (state를 버림)
x = torch.where(do, resample, x)             # 현재 (state를 보존)
```
이는 CTMC 전이 커널이 아니라 **비-마르코프적 재노이즈**로, $p_t$ 마진을 보존하지 않는다.
따라서 이득은 "Euler보다 정확해서"가 아니라 **"올바른 CTMC 커널을 시뮬레이션하게 되어서"** 이다.
문서·발표에서는 이렇게 서술해야 한다.

---

## 8. 대조군 — CFM (continuous relaxation)

Baseline으로 유지 중인 연속 완화 버전:

$$
s = 2b - 1 \in \{-1,+1\}^D, \qquad
x_0 \sim \mathcal{N}(0, I_D), \qquad
x_t = (1-t)\,x_0 + t\,s,
$$
$$
u_t = s - x_0, \qquad
\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t,\,x_0,\,(b,y)}\big\|\, v_\theta(x_t, t, y) - u_t \,\big\|^2,
$$
샘플링: Euler ODE $\dot x = v_\theta(x,t,y)$, $t: 0\to1$, 이후 $\hat b = \mathbb{1}[x_1 > 0]$.

| | CFM | DFM |
|---|---|---|
| 상태공간 | $\mathbb{R}^D$ (완화) | $\mathcal{S}^{T}$ (이산) |
| source | $\mathcal{N}(0,I)$ | $\mathrm{Unif}(\mathcal{S})^{\otimes T}$ |
| 예측 대상 | velocity $u_t$ | $x_1$-posterior |
| loss | MSE | (B)CE |
| 이산화 시점 | **마지막에 threshold** | **매 스텝** (항상 이산) |
| grouping $g$ | 단순 reshape (수학 불변) | **$V=2^g$ 를 바꿈 (본질적)** |

CFM에서는 $g$ 가 tensor shape만 바꾸므로 설계 변인이 아니다. DFM에서만 $g$ 가 어휘를 정의한다.

---

## 9. 전체 알고리즘 요약

**Training**
1. $(b, y) \sim q$, $x_1 = \phi_g(b)$
2. $t \sim \mathcal{U}[0,1]$
3. $x_t \sim p_t(\cdot\mid x_1)$: 각 토큰 독립, 확률 $t$ 로 유지 / $1-t$ 로 $\mathrm{Unif}(\mathcal{S})$
4. 확률 $p_{\text{drop}}$ 로 $y \leftarrow \varnothing$
5. $\ell = f_\theta(x_t, t, y)$, loss $= \mathrm{CE}(\ell, x_1)$

**Sampling** ($N$ steps, guidance $s$)
1. $x \sim \mathrm{Unif}(\mathcal{S})^{\otimes T}$
2. for $i = 0,\dots,N-1$: $\;t = i/N$
   - $\ell = \ell_\varnothing + s(\ell_y - \ell_\varnothing)$, $\;\hat p = \mathrm{softmax}(\ell)$
   - $a = 1/(N-i)$
   - 각 토큰 독립: w.p. $a$ 는 $\hat p$ 에서 재추출(마지막 스텝은 $\arg\max$), w.p. $1-a$ 는 유지
3. return $b = \psi_g(x)$

---

## 10. 평가 (evaluation)

| 지표 | 정의 | 성격 |
|---|---|---|
| **BitAcc** | $\frac1D\sum_j \mathbb{1}[\hat b_j = b^{\text{GT}}_j]$ | sanity |
| **PatAcc** | $\mathbb{1}[\hat b = b^{\text{GT}}]$ 의 평균 | sanity (엄격) |
| **Surrogate err** | $\big\|\,\mathcal{F}_{\text{surr}}(\hat b) - y\,\big\|$ | **주 지표** |

§0에서 언급했듯 $\mathcal{F}^{-1}(y)$ 는 다중해이므로 BitAcc/PatAcc는 **유효한 대안해를 부당하게 penalize**한다.
이들은 학습이 붕괴하지 않았는지 확인하는 용도로만 쓰고, **스펙트럼 재현 오차를 주 지표**로 삼는다.

---

## 11. 미해결 / 미탐색

- $\eta > 0$ (stochasticity/corrector) — §6.5, 미구현
- 비선형 $\kappa_t$ — §7.3에 따르면 exact 커널의 이득이 여기서만 실현됨
- $t$-weighting $w(t) \ne 1$
- 참 posterior의 factorization error를 직접 측정하는 진단
- $g$ 와 스텝 수 $N$ 의 trade-off (이론상 $g\uparrow$ ⇒ 필요한 $N\downarrow$)

---

## 기호 표

| 기호 | 의미 |
|---|---|
| $b \in \{0,1\}^D$ | binary antenna mask, $D=100$ |
| $y \in \mathbb{R}^P$ | 목표 스펙트럼 (condition), $P=201$ |
| $g,\; V=2^g,\; T=D/g$ | group size, 어휘 크기, 토큰 수 |
| $\mathcal{S}=\{0..V-1\},\; \mathcal{X}=\mathcal{S}^T$ | 심볼 집합, 상태공간 |
| $\phi_g,\psi_g$ | bits↔tokens 전단사 |
| $\kappa_t$ | 노이즈 스케줄 (현재 $=t$) |
| $p_t(\cdot\mid x_1)$ | conditional probability path |
| $u_t(v,z)$ | rate matrix (generator) |
| $\lambda_t = \dot\kappa_t/(1-\kappa_t)$ | 점프 rate |
| $\hat p = p_\theta(x_1\mid x_t,t,y)$ | 예측 posterior |
| $a = (\kappa_{t'}-\kappa_t)/(1-\kappa_t)$ | 스텝 전이(재추출) 확률 |
| $s$ | CFG guidance scale |
