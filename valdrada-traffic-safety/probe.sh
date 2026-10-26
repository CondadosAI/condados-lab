#!/bin/bash
# P0 check 1: does the Valdrada server render on the RTX 5060 under Xvfb in a container, or fall back to SwiftShader?
# Usage: [MODE=xvfb|xorg|vgl] [BUDGET="<budget.py args>"] [CHECKS=1] [RECORDS="<record.py args>|..."] [SCRIPT="<script.py args>"] bash probe.sh <variant> "<chromium flags>"
# xorg: a headless NVIDIA Xorg on :1 instead of Xvfb; vgl: Xvfb with GL redirected to the GPU by VirtualGL (EGL back end).   Markers: logs/probe_<variant>_DONE / _FAILED.
# The GPU processes at launch are the baseline; a new foreign compute process stops OUR container only.
set -uo pipefail
V=$1; FLAGS=${2:-}; MODE=${MODE:-xvfb}
F=${VALDRADA_WORK:-$PWD}; O=$F/logs; C=valdrada-gpu; N=probe_$V
rm -f $O/${N}_DONE $O/${N}_FAILED
log() { echo "$(date -u +%T) $*" >> $O/$N.batch.log; }
ours() { docker top $C -eo pid 2>/dev/null | tail -n +2 | tr -d ' '; }
gpu_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | sort; }
BASE=$(gpu_pids)
foreign() { comm -23 <(gpu_pids) <( (echo "$BASE"; ours) | sort -u); }
X() { docker exec -w /f/src $C bash -c "$1"; }
[ -z "$(foreign)" ] || { log "GPU busy: $(foreign)"; echo busy > $O/${N}_FAILED; exit 1; }
docker start $C >/dev/null
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,clocks.max.sm,power.draw \
  --format=csv,noheader -l 2 > $O/${N}_gpu.csv &
SAMPLER=$!
( while sleep 5; do f=$(foreign); [ -z "$f" ] && continue
    log "someone else started on the GPU ($f): stopping our container"
    echo "yielded to $f" > $O/${N}_FAILED; docker stop -t 5 $C >/dev/null; exit 0; done ) &
WD=$!
cleanup() { docker exec -u 0 $C bash -c "pkill -f [e]lectron; pkill -f [v]ite; pkill -f [X]vfb; pkill -f [X]org" >/dev/null 2>&1; kill $WD $SAMPLER 2>/dev/null; }
trap cleanup EXIT
step() { name=$1; shift; t0=$(date +%s); log "start $name"
  X "$*" > $O/${N}_$name.log 2>&1 || { log "FAILED $name"; echo "$name" > $O/${N}_FAILED; exit 1; }
  log "done $name in $(( $(date +%s) - t0 )) s"; [ -f $O/${N}_FAILED ] && exit 1; }
log "flags: $FLAGS"
step stamp "git rev-parse HEAD; git status --short | head"
docker exec -d -w /f/src/client $C bash -c "npm run dev -- --host 127.0.0.1 > /f/logs/${N}_vite.log 2>&1"
step vite "for i in \$(seq 60); do curl -sf http://127.0.0.1:5219 >/dev/null && exit 0; sleep 2; done; exit 1"
XV="xvfb-run -a -s '-screen 0 1920x1080x24'"
case $MODE in
  xorg) docker exec -u 0 -d $C bash -c "Xorg :1 -config /f/docker/xorg-nvidia.conf -noreset -nolisten tcp -novtswitch -sharevts -ac +extension GLX -logfile /f/logs/${N}_Xorg.log > /f/logs/${N}_Xorg.out 2>&1"
        step xorg "for i in \$(seq 30); do DISPLAY=:1 xdpyinfo >/dev/null 2>&1 && DISPLAY=:1 glxinfo -B; [ \$? = 0 ] && exit 0; sleep 1; done; exit 1"
        RUN="DISPLAY=:1" ;;
  vgl)  step vgl "$XV /opt/VirtualGL/bin/vglrun -d egl glxinfo -B"
        RUN="$XV /opt/VirtualGL/bin/vglrun -d egl" ;;
  *)    RUN="$XV" ;;
esac
docker exec -d -w /f/src/server -e XDG_RUNTIME_DIR=/tmp $C bash -c "$RUN ./node_modules/electron/dist/electron --no-sandbox $FLAGS . --dev-url=http://127.0.0.1:5219 --headless --pedestrians procedural --res 1280x720 --verbose > /f/logs/${N}_server.log 2>&1"
step ready "/f/home/venv/bin/python -c 'import valdrada; c=valdrada.Client(timeout=300); c.wait_until_ready(timeout=600); print(c.get_server_info())'"
if [ -n "${CHECKS:-}" ]; then
  PY="env PYTHONUNBUFFERED=1 /f/home/venv/bin/python"
  step check2 "cd PythonAPI/examples/tutorials && $PY record_dataset.py --frames 24 --out /f/out/${N}_check2"
  step check3 "timeout 2400 $PY /f/check3.py --out /f/out/${N}_check3"
  X "pkill -f [e]lectron" >/dev/null 2>&1; sleep 5                 # free the 5.8 GB the renderer holds
  step zeroshot "timeout 1800 $PY /f/zeroshot.py /f/out/${N}_check2 /f/out/${N}_check3 /f/out/probe_B1_budget --model small --out /f/out/${N}_zeroshot.json"
elif [ -n "${RECORDS:-}" ]; then                 # "args1|args2|...": one record.py run each
  IFS='|' read -ra RECS <<< "$RECORDS"; k=0
  for r in "${RECS[@]}"; do k=$((k+1)); step record$k "cd /f && env PYTHONUNBUFFERED=1 timeout 1800 /f/home/venv/bin/python record.py $r"; done
elif [ -n "${SCRIPT:-}" ]; then                  # one script from /f, e.g. SCRIPT="pole_compare.py --out /f/out/x"
  step script "cd /f && env PYTHONUNBUFFERED=1 timeout 1800 /f/home/venv/bin/python $SCRIPT"
elif [ -n "${BUDGET:-}" ]; then
  step budget "PYTHONUNBUFFERED=1 timeout 2400 /f/home/venv/bin/python /f/budget.py $BUDGET --out /f/out/${N}_budget"
  nvidia-smi > $O/${N}_nvsmi.txt
else
step bench1 "timeout 1500 /f/home/venv/bin/python PythonAPI/examples/benchmark.py --ticks 60"
nvidia-smi > $O/${N}_nvsmi.txt
step bench2 "timeout 1500 /f/home/venv/bin/python PythonAPI/examples/benchmark.py --ticks 60"
step quick "timeout 900 /f/home/venv/bin/python PythonAPI/examples/quickstart.py --frames 20 --out /f/out/${N}_quickstart"
fi
cleanup; log "GPU given back"
touch $O/${N}_DONE
