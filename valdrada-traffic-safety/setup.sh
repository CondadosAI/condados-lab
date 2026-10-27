#!/bin/bash
# CPU setup: npm dependencies and the v0.5.2 binary banks. Markers: logs/setup_DONE or logs/setup_FAILED.
set -uo pipefail
F=${VALDRADA_WORK:-$PWD}; O=$F/logs; rm -f $O/setup_DONE $O/setup_FAILED
X() { docker exec -w /f/src valdrada-gpu bash -c "$1"; }
step() { n=$1; shift; echo "$(date -u +%T) start $n" >> $O/setup.batch.log
  X "$*" > $O/setup_$n.log 2>&1 || { echo "$(date -u +%T) FAILED $n" >> $O/setup.batch.log; echo $n > $O/setup_FAILED; exit 1; }
  echo "$(date -u +%T) done $n" >> $O/setup.batch.log; }
step npm_root "npm install --no-audit --no-fund"
step npm_client "cd client && npm install --no-audit --no-fund"
step npm_server "cd server && npm install --no-audit --no-fund && (test -d node_modules/electron/dist || node node_modules/electron/install.js) && ls node_modules/electron/dist | head"
step banks "hf download mehmetkeremturkcan/valdrada --repo-type dataset --revision v0.5.2 --include 'Content/tiles/*' --include 'Content/models/*' --include 'Content/textures/*' --local-dir /f/home/hfcache && mv /f/home/hfcache/Content/tiles /f/home/hfcache/Content/models /f/home/hfcache/Content/textures client/public/ && du -sh client/public/tiles client/public/models client/public/textures"
step pyapi "python3 -m venv /f/home/venv --system-site-packages && /f/home/venv/bin/pip install -q -e PythonAPI && /f/home/venv/bin/python -c 'import valdrada; print(valdrada.__file__)'"
touch $O/setup_DONE
