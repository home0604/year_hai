#!/bin/bash
# Eval sweep over sampling steps x CFG scale for the 2026-07-09 checkpoints.
#   steps: 1 2 4 10 15 20 50
#   cfg  : DFM {1,1.5,2,3,5} / CFM {1,1.5,2,3} / *_nocfg {1.0} only
# *_nocfg 모델은 drop_prob=0 으로 학습되어 null_y_embed 에 gradient 가 간 적이 없음
# → forward_uncond 가 무의미하므로 gs=1.0 에서만 평가한다.
set -u
cd "$(dirname "$0")"
source /hai/anaconda3/etc/profile.d/conda.sh
conda activate timing

EXP=experiments/20260709
OUT=experiments/20260709/eval_sweep
mkdir -p "$OUT"
CSV="$OUT/sweep.csv"
STEPS="1 2 4 10 15 20 50"
COMMON="--arch stable3dit --sample_steps $STEPS --topk 10 --topk_eval_n 2000 \
        --acc_eval_n 4096 --sample_batch 512 --csv $CSV --seed 0"

run () {  # run <gpu> <tag> <extra args...>
  local gpu=$1 tag=$2; shift 2
  echo "[GPU $gpu] $tag"
  CUDA_VISIBLE_DEVICES=$gpu python -u eval.py \
      --ckpt "$EXP/$tag/best_model.pth" --tag "$tag" $COMMON "$@" \
      > "$OUT/$tag.log" 2>&1 &
}

run 4 dfm_g1       --flow_type dfm --group_size 1            --guidance_scale 1.0 1.5 2.0 3.0 5.0
run 5 dfm_g4       --flow_type dfm --group_size 4 --spatial  --guidance_scale 1.0 1.5 2.0 3.0 5.0
run 6 cfm_g1       --flow_type cfm --group_size 1            --guidance_scale 1.0 1.5 2.0 3.0
run 7 dfm_g1_nocfg --flow_type dfm --group_size 1            --guidance_scale 1.0
run 0 dfm_g4_nocfg --flow_type dfm --group_size 4 --spatial  --guidance_scale 1.0
run 2 cfm_g1_nocfg --flow_type cfm --group_size 1            --guidance_scale 1.0

wait
echo "=== done -> $CSV ==="
