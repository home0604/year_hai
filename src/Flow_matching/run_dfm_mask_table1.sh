#!/bin/bash
# dfm_mask_g4 (absorbing-start, 20260726) 를 시간예산 프로토콜로 Table1 sweep 에 추가.
#   목적: uniform-start dfm_g4 vs absorbing-start dfm_mask_g4 를 dual 에서 비교(시작 regime 가설 검증).
#   steps{4,10,20}, GPU 2·5 분배. 출력은 table1_out (기존 dfm_g4/ar 과 같은 집계).
set -u
cd /hai/home/lsh/antenna/year_hai/src/Flow_matching
OUT=table1_out; T=300; NS=20; ND=10
PLOG=$OUT/_progress_mask.log

run () { # gpu, configs...
  local gpu=$1; shift
  for cfg in "$@"; do
    echo "[$(date +%m-%d\ %H:%M:%S)] GPU$gpu ▶ $cfg" >> $PLOG
    python -u table1_search.py --config "$cfg" --gpu "$gpu" --T $T \
        --n_single $NS --n_dual $ND --out $OUT > $OUT/$cfg.log 2>&1
    echo "[$(date +%m-%d\ %H:%M:%S)] GPU$gpu ✔ $cfg" >> $PLOG
  done
}

run 2 dfm_mask_g4_s4 dfm_mask_g4_s10 &
run 5 dfm_mask_g4_s20 &
wait
echo "[$(date +%m-%d\ %H:%M:%S)] dfm_mask DONE" >> $PLOG
