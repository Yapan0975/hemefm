#!/bin/bash
set -e
cd ~/hemefm
source .venv/bin/activate
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=1,2

for LR in 1e-4 5e-4 1e-3; do
  CFG="finetune_b3_lr_${LR//[.-]/_}"
  LOG="logs/b3-lr-sweep/${CFG}.log"
  mkdir -p "$(dirname $LOG)"
  echo "=== [$(date +%H:%M:%S)] Starting ${CFG} ===" | tee -a logs/b3-lr-sweep/sweep.log
  python -m hemefm.train experiment=${CFG} > $LOG 2>&1
  echo "=== [$(date +%H:%M:%S)] Completed ${CFG} ===" | tee -a logs/b3-lr-sweep/sweep.log
done
echo '=== B3 lr-sweep ALL DONE ===' | tee -a logs/b3-lr-sweep/sweep.log
