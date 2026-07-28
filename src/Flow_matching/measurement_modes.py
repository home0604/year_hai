# -*- coding: utf-8 -*-
"""
측정방식(measurement mode) — 모델 간 성능을 비교할 때 무엇을 재고 있는지 구분한다.

배경:
  AR 학습 코드(train_ordering_no_sample_images.py:300-321, :342-360)는 BitAcc/PatAcc 를
  **teacher forcing** 으로 잰다 — 위치 k 에서 정답 prefix b_<k 를 넣고 b_k 를 예측한다.
  wandb 에 찍히는 AR 의 BitAcc 가 이 값이다.

  반면 eval.py 가 재는 DFM/CFM 의 BitAcc 는 **실제로 샘플링한 결과**다.

  이 둘은 다른 지표다. 나란히 놓으면 안 된다.
  (실측: AR teacher-forced BitAcc 0.869 vs AR 자유생성 BitAcc 0.542)

교정 방향:
  FM 쪽에 teacher-forcing 대응물을 만드는 것이 아니다 — FM 의 측정방식은 원래 옳다.
  AR 을 **자유생성으로 다시 재는 것**이 옳다. 이 모듈은 그 자유생성 디코더를 제공한다.
"""
import torch

BOS_IDX = 2


# ===========================================================================
# 생성(free-running) — 모델이 자기 출력만 보고 처음부터 만든다. 공정 비교의 기준.
# ===========================================================================

@torch.no_grad()
def free_running_ar(model, y, num_bits, temperature=1.0, greedy=False, bos_idx=BOS_IDX):
    """AR ancestral decoding. NFE = num_bits (순차).

    학습 코드에는 이 루프가 없다(teacher forcing 만 있음) — 여기서 구현한다.
    """
    B = y.size(0)
    tokens = torch.full((B, 1), bos_idx, dtype=torch.long, device=y.device)
    for _ in range(num_bits):
        p = torch.sigmoid(model(y, tokens)[:, -1] / temperature)
        nxt = (p > 0.5).long() if greedy else (torch.rand_like(p) < p).long()
        tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
    return tokens[:, 1:]                                   # BOS 제거


@torch.no_grad()
def free_running_fm(model, y, num_bits, steps, guidance_scale,
                    flow_type, group_size=1, flip_pred=False):
    """DFM/CFM 생성. NFE = steps (병렬). eval.py 의 sample_fn 과 동일하다."""
    from flow_matching import sample_fm_cfg, sample_dfm_cfg
    if flow_type == "cfm":
        return sample_fm_cfg(model, y, num_bits, steps, guidance_scale,
                             group_size=group_size)
    return sample_dfm_cfg(model, y, num_bits, steps, guidance_scale,
                          group_size=group_size, flip_pred=flip_pred)


# ===========================================================================
# teacher forcing — AR 에만 존재한다. 생성 지표가 아니라 학습 진단이다.
# ===========================================================================

@torch.no_grad()
def teacher_forced_ar(model, y, x_gt, bos_idx=BOS_IDX):
    """wandb 의 AR BitAcc/PatAcc 를 그대로 재현한다.

    이 함수는 **비교용이 아니라 대조용**이다 — 같은 체크포인트에서 이 값과
    free_running_ar 의 값이 얼마나 벌어지는지를 보여주기 위해 존재한다.
    """
    B = x_gt.size(0)
    gt = x_gt.long()
    bos = torch.full((B, 1), bos_idx, dtype=torch.long, device=x_gt.device)
    tokens_in = torch.cat([bos, gt[:, :-1]], dim=1)        # ← 정답 prefix
    pred = (torch.sigmoid(model(y, tokens_in)) > 0.5).long()
    return (pred == gt).float().mean().item(), (pred == gt).all(dim=1).float().mean().item()


# ===========================================================================
# 구조적 진단: 데이터 support (정확히 60 개의 1) 준수율
# ===========================================================================

def support_rate(bits, target_ones=60):
    """생성 마스크가 데이터 support 안에 있는 비율.

    AR 은 chain rule 이라 이미 놓은 비트를 세면서 진행할 수 있어 ≈1.0 이 나온다.
    DFM 은 좌표별 독립 사후분포라 셀 수 없다 — 이 격차가 곧 factorization error 다.
    """
    n1 = bits.float().sum(-1).reshape(-1)
    return {
        "ones_mean": n1.mean().item(),
        "ones_std": n1.std().item(),
        "support_rate": (n1 == target_ones).float().mean().item(),
    }
