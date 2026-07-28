#!/bin/bash
# =============================================================================
# Flow Matching DiT 실험 스크립트
# 10x10 patch-antenna inverse design
#
# 사용법:
#   conda activate timing
#   cd /hai/home/lsh/antenna/year_hai/src/Flow_matching
#   bash run_experiments.sh
#
# 결과: experiments/YYYYMMDD/<exp_name>/
# =============================================================================

DATA="/hai/home/lsh/antenna/year_hai/data/datasets"
COMMON="--data_root $DATA --project Flow_antenna --ema_decay 0 --arch stable3dit --d_model 512 --num_layers 15 --nhead 4 --dim_ff 768 --epochs 500 --batch_size 128 --lr 1e-4"

# DFM g4 (spatial, CFG on)
CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --exp_name dfm_g4         --flow_type dfm --group_size 4 --spatial              $COMMON &


# DFM g1 (CFG on)
CUDA_VISIBLE_DEVICES=4 nohup python -u train.py --exp_name dfm_g1         --flow_type dfm --group_size 1                       $COMMON &


# CFM g1 (CFG on)
CUDA_VISIBLE_DEVICES=6 nohup python -u train.py --exp_name cfm_g1         --flow_type cfm --group_size 1                       $COMMON &


CUDA_VISIBLE_DEVICES=7 nohup python -u train.py --exp_name cfm_g1_xpred --cfm_param xpred_ste --lambda_ste 0.0 --flow_type cfm --group_size 1 $COMMON &

CUDA_VISIBLE_DEVICES=3 nohup python -u train.py --exp_name dfm_mask_g1 --flow_type dfm_mask --group_size 1 $COMMON &

CUDA_VISIBLE_DEVICES=5 nohup python -u train.py --exp_name dfm_mask_g4 --flow_type dfm_mask --group_size 4 --spatial $COMMON &

echo "All experiments launched. Check experiments/$(date +%Y%m%d)/"
