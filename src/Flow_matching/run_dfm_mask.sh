#!/bin/bash
# =============================================================================
# Absorbing (mask) DFM 실험 — g1 / g4 (CFG on) 두 종.
#   · absorbing 은 collision 이 없어 grouping 의 주 동기가 사라진다
#     → g1 이 이론적 기본. g4 는 패치 내 결합분포 표현력 확인용.
#   · 나머지(cfm, uniform dfm)는 이미 학습 완료. nocfg 는 생략.
#
#   conda activate timing
#   cd /hai/home/lsh/antenna/year_hai/src/Flow_matching
#   bash run_dfm_mask.sh
#
# 결과: experiments/YYYYMMDD/<exp_name>/   (GPU 0~1 사용; 4~7 은 xpred 실행 중)
# =============================================================================

export WANDB_MODE=online          # offline 로 빠지지 않도록 명시 (Flow_antenna 프로젝트에 실시간 로깅)

DATA="/hai/home/lsh/antenna/year_hai/data/datasets"
COMMON="--data_root $DATA --project Flow_antenna --ema_decay 0 --arch stable3dit --d_model 512 --num_layers 15 --nhead 4 --dim_ff 768 --epochs 500 --batch_size 128 --lr 1e-4"

# DFM-mask g4 (spatial, CFG on) — 표현력 비교용
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --exp_name dfm_mask_g4 --flow_type dfm_mask --group_size 4 --spatial $COMMON > dfm_mask_g4.log 2>&1 &

# DFM-mask g1 (CFG on) — absorbing 의 이론적 기본
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --exp_name dfm_mask_g1 --flow_type dfm_mask --group_size 1           $COMMON > dfm_mask_g1.log 2>&1 &

echo "dfm_mask g4, g1 launched (GPU 0, 1). 결과: experiments/$(date +%Y%m%d)/"
echo "로그: dfm_mask_g4.log / dfm_mask_g1.log"
