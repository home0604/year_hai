# Flow Matching — Research Notes (running)

Binary antenna inverse design (DFM / CFM) 연구노트 + 실험결과를 날짜별로 누적.
최신 날짜가 위.

**기록 방법**: 실험/변경 있을 때마다 맨 위에 `## YYYY-MM-DD` 블록 추가 → 목차에 한 줄 링크.
지표 정의는 맨 아래 [지표 정의](#지표-정의) 참조.

**관련 문서**
- [`FORMULATION.md`](./FORMULATION.md) — 수학적 정식화 (state/vocab/path/rate/sampler 유도)
- [`DESIGN_CHOICES.md`](./DESIGN_CHOICES.md) — 설계 변인 결정 트리·표
- [`research_note_20260705.md`](./research_note_20260705.md) — DFM 학습실패 원인분석 상세 (2026-07-05)

---

## 목차

- [2026-07-07](#2026-07-07) — exact 유한전이 sampler 교체, 설계 문서 정리
- [2026-07-05](#2026-07-05) — DFM binary 학습 실패 분석, grouping/OT 도입
- [지표 정의](#지표-정의)

---

## 2026-07-07

**요약**: DFM 샘플러를 재샘플 방식 → **exact 유한전이(state-preserving)**로 교체. 설계 변인 문서화.

### 변경 사항
- `sample_dfm_cfg`: 매 스텝 전체 재샘플(현재상태 버림) → **per-bit 확률 `a=(t'-t)/(1-t)`로만 예측 쪽 점프, 나머지 유지**. binary/grouped 양쪽 적용.
- 학습(`dfm_loss`)은 불변 → 기존 체크포인트 재학습 없이 샘플러만 교체 가능.
- `DESIGN_CHOICES.md` 신규 (결정 트리 + 변인표 + CTMC/DFM + 평가기준).
- `flip_pred`는 기본 off 유지 (binary 본선에선 no-op).

### ⚠️ 정정 (2026-07-12)
"exact 유한전이가 Euler/tau-leaping의 상위호환"이라는 초기 서술은 **틀렸다**.
- exact: $a = (\kappa_{t'}-\kappa_t)/(1-\kappa_t)$, Euler: $a \approx \dot\kappa_t\Delta t/(1-\kappa_t)$.
- **linear $\kappa_t=t$ 에서는 둘이 정확히 일치** → 현재 설정에서 수치적으로 동일.
- 정확적분의 실익은 **비선형 $\kappa_t$** (cosine 등)에서만 발생.
- 이번 교체의 실제 이득: 이전 샘플러가 $x_t$를 버리고 uniform 재노이즈하던
  (CTMC 전이 커널이 아닌, $p_t$ 마진을 보존하지 않는) 방식이었던 것을 고친 것.

### 결과

| Model | Arch | Flow | gs | V | Epoch | BitAcc | PatAcc | Surrogate err | Sampler | Notes |
|-------|------|------|----|----|-------|--------|--------|---------------|---------|-------|
| _(fill)_ | | | | | | | | | exact-transition | |

### 관찰 / TODO
- [ ] few-step에서 이전 재샘플 대비 개선 여부 확인
- [ ] steps / guidance_scale 재스윕 (샘플러 바뀌어 최적값 이동 가능)
- [ ] 비선형 $\kappa_t$ (cosine) 도입 시에만 exact 커널의 이득 검증 가능

---

## 2026-07-05

**요약**: DFM binary(V=2) 학습 실패(BitAcc~0.54) 원인 = weak corruption(collision 50%).
해결책으로 bit grouping, OT coupling 도입. → 상세: [`research_note_20260705.md`](./research_note_20260705.md)

### 핵심 발견
- binary vocab에서 uniform 교체 시 collision 50% → corrupted vs clean 구분 불가, gradient signal 미약.
- **Bit grouping** (gs bits → 1 token, V=2^gs)으로 collision↓ / position당 정보밀도↑.
- **OT coupling** (Hamming Hungarian): x₀→x₁ 평균 Hamming 25.1→19.0 (24% 감소).
- 문헌: CoBit(2026)가 naive bit-level 실패 + grouping 유효를 직접 실증.

### 진행 실험 (wandb: `rogers_inverse_dit`)

| GPU | Arch | Flow | gs | V | BitAcc | PatAcc | Surrogate err | Notes |
|-----|------|------|----|----|--------|--------|---------------|-------|
| 4 | SmallDiT | DFM | 4 | 16 | _(fill)_ | | | |
| 5 | SmallDiT | DFM | 10 | 1024 | | | | |
| 6 | Stable3DiT | DFM | 4 | 16 | | | | |
| 7 | SmallDiT | CFM | 4 | 16 | | | | |

### 이전 실패 baseline (참고)

| Model | gs | V | Epoch | BitAcc | PatAcc |
|-------|----|----|-------|--------|--------|
| SmallDiT (32.9M) | 1 | 2 | 140/500 | 0.54 | 0.00 |
| Stable3DiT (76.8M) | 1 | 2 | 45/500 | 0.54 | 0.00 |

---

## 지표 정의

| 지표 | 정의 | 성격 |
|---|---|---|
| **BitAcc** | 생성 mask vs GT mask per-bit 일치율 | 재현/sanity |
| **PatAcc** | 전체 패턴 exact match 비율 (all bits) | 재현/sanity (엄격) |
| **Surrogate err** | \|surrogate(x_gen) − y\| — forward 모델 스펙트럼 재현 오차 | **주 지표** |

**caveat**: inverse design은 one-to-many라 BitAcc/PatAcc는 유효 대안 해를 penalize.
→ sanity로만 보고, **surrogate 스펙트럼 오차를 주 지표**로. (`eval.py` 참조)
