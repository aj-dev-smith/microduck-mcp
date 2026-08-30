#!/bin/bash
# Duck cam: live viser 3D viewer beside a training run on the GPU box.
#
# Runs ON THE BOX (inside WSL Ubuntu — copy scripts/box/* to ~ via
# `scp scripts/box/* duck-4090-wsl:`), in its own tmux session so it
# survives disconnects. The viewer loads the newest checkpoint of the
# newest run; the checkpoint panel in the browser UI hot-loads later
# ones while training continues.
#
# From the Mac:  ssh -f -N -L 8080:localhost:8080 duck-4090-wsl
# then open     http://localhost:8080
#
# Usage: launch_viewer.sh [task-id]   (default: Mjlab-StandUp-Flat-MicroDuck)

TASK="${1:-Mjlab-StandUp-Flat-MicroDuck}"
LOGS=~/microduck_rl/logs/rsl_rl

RUN_DIR=$(ls -td "$LOGS"/*/*/ 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ]; then
  echo "no run dirs under $LOGS" >&2
  exit 1
fi
LATEST=$(ls -t "$RUN_DIR"model_*.pt 2>/dev/null | head -1)
if [ -z "$LATEST" ]; then
  echo "no model_*.pt in $RUN_DIR yet — wait for the first checkpoint" >&2
  exit 1
fi

tmux kill-session -t duck-viewer 2>/dev/null
tmux new-session -d -s duck-viewer \
  "cd ~/microduck_rl && ~/.local/bin/uv run python ~/play_viser_patched.py $TASK --viewer viser --num-envs 1 --checkpoint-file $LATEST 2>&1 | tee ~/viewer.log"
sleep 2
tmux ls
echo "VIEWER-LAUNCHED task=$TASK checkpoint=$LATEST"
