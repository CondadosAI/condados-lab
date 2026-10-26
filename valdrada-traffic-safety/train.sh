#!/bin/bash
# Fine-tune RF-DETR Small on the P0 frames, split by capture run: train = budget poles (B1), valid = author's
# record_dataset pose (C2 check2), test = check3 (pole-search pose clip + conditions + Broadway & W 96th).
# Scores zero-shot and fine-tuned on the same test split.   Markers: logs/train_<V>_DONE / _FAILED.
set -uo pipefail
V=${1:-T1}; EPOCHS=${EPOCHS:-40}; RUN=${RUN:-$V}   # RUN: the fine-tune folder to score (EVAL_ONLY)
F=${VALDRADA_WORK:-$PWD}; O=$F/logs; C=valdrada-gpu; N=train_$V
rm -f $O/${N}_DONE $O/${N}_FAILED
log() { echo "$(date -u +%T) $*" >> $O/$N.batch.log; }
ours() { docker top $C -eo pid 2>/dev/null | tail -n +2 | tr -d ' '; }
gpu_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | sort; }
BASE=$(gpu_pids)
foreign() { comm -23 <(gpu_pids) <( (echo "$BASE"; ours) | sort -u); }
X() { docker exec -w /f -e PYTHONUNBUFFERED=1 -e HOME=/f/home $C bash -c "$1"; }
[ -z "$(foreign)" ] || { log "GPU busy: $(foreign)"; echo busy > $O/${N}_FAILED; exit 1; }
docker start $C >/dev/null
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,clocks.max.sm,power.draw \
  --format=csv,noheader -l 5 > $O/${N}_gpu.csv &
SAMPLER=$!
( while sleep 5; do f=$(foreign); [ -z "$f" ] && continue
    log "someone else started on the GPU ($f): stopping our container"
    echo "yielded to $f" > $O/${N}_FAILED; docker stop -t 5 $C >/dev/null; exit 0; done ) &
WD=$!
trap "kill $WD $SAMPLER 2>/dev/null" EXIT
step() { name=$1; shift; t0=$(date +%s); log "start $name"
  X "$*" > $O/${N}_$name.log 2>&1 || { log "FAILED $name"; echo "$name" > $O/${N}_FAILED; exit 1; }
  log "done $name in $(( $(date +%s) - t0 )) s"; [ -f $O/${N}_FAILED ] && exit 1; }
PY=/f/home/venv/bin/python; D=/f/out/coco_$RUN; R=/f/out/ft_$RUN
step stamp "md5sum to_coco.py finetune.py zeroshot.py train.sh"
TEST=/f/out/probe_C2_check3                     # scored from the source frames: the scorer needs their instance JSON
if [ -z "${EVAL_ONLY:-}" ]; then                # EVAL_ONLY=1 rescores an existing run's checkpoint
  step coco "$PY to_coco.py --train /f/out/probe_B1_budget --valid /f/out/probe_C2_check2 --test $TEST --out $D"
  step finetune "$PY finetune.py --dataset $D --out $R --epochs $EPOCHS"
fi
step eval_ft "$PY zeroshot.py $TEST --weights $R/checkpoint_best_total.pth --out $R/eval_finetuned.json"
step eval_zs "$PY zeroshot.py $TEST --out $R/eval_zeroshot.json"
kill $WD 2>/dev/null; docker stop -t 5 $C >/dev/null; log "GPU given back"
touch $O/${N}_DONE
