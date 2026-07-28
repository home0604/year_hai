#!/bin/bash
# =============================================================================
# 논문 Table I / Fig.16 재현 sweep — 시간예산 탐색(T=5분) × 17 config.
#   config = ar(MAB) + {dfm_g4,dfm_g1,cfm_g1,cfm_g1_xpred} × steps{1,4,10,20}
#   target = single-20 + dual-10 (table1_out/table1_*_idx.npy)
#   GPU 1 한 장만 사용(0은 비워둠). config당 30 target × 300s ≈ 2.5h → 총 ~43h.
#
#   실행 순서: AR → (step1 4종) → (step4 4종) → (step10 4종) → (step20 4종)
#     같은 step끼리 묶여 끝나므로 중간 결과로도 모델 간 비교가 바로 가능.
#
#   conda activate timing
#   cd /hai/home/lsh/antenna/year_hai/src/Flow_matching
#   bash run_table1.sh
# =============================================================================
set -u
cd /hai/home/lsh/antenna/year_hai/src/Flow_matching
OUT=table1_out
GPU=1
T=300
NS=20
ND=10

# 모델 3종(dfm_g4/dfm_g1/cfm_g1_xpred) × steps{4,10,20}. step1 은 MSE 너무 커서 제외,
# cfm_g1(velocity) 제외. ar 은 이미 완료(재실행 안 함). step 별로 묶어 중간비교 가능.
QUEUE=(
  dfm_g4_s4 dfm_g1_s4 cfm_g1_xpred_s4
  dfm_g4_s10 dfm_g1_s10 cfm_g1_xpred_s10
  dfm_g4_s20 dfm_g1_s20 cfm_g1_xpred_s20
)

for cfg in "${QUEUE[@]}"; do
  echo "[$(date +%m-%d\ %H:%M:%S)] GPU$GPU ▶ $cfg" >> $OUT/_progress.log
  python -u table1_search.py --config "$cfg" --gpu "$GPU" --T $T \
      --n_single $NS --n_dual $ND --out $OUT > $OUT/$cfg.log 2>&1
  echo "[$(date +%m-%d\ %H:%M:%S)] GPU$GPU ✔ $cfg" >> $OUT/_progress.log
done
echo "[$(date +%m-%d\ %H:%M:%S)] ALL DONE" >> $OUT/_progress.log
