#!/bin/bash
# Full rough-terrain baseline, 2000 epochs (E003).
#
#   configs/baseline_rough_2000.json = configs/ablation_baseline.json
#   (AttnRes, 2x [4,128], attn_res_heads: 4, scene_rough.xml, depth camera on)
#   with num_epochs 500 -> 2000 and save_interval 50 -> 100. Nothing else differs,
#   so the 500-epoch ablation matrix stays comparable to its first 500 epochs.
#
# One arm at a time on GPU 0 (RTX 3070, 8 GB): a second concurrent run at
# --vec_env_nums 16 OOMs. ~70 s/epoch measured on rough terrain -> ~39 h.
#
# Runs inside a detached tmux session so it survives closing the terminal and
# can be attached to at any time.
#
# Launch:  ./run_baseline_rough_2000.sh
# Attach:  tmux attach -t baseline2000      (detach with Ctrl-b d)
# Watch:   tail -f log_baseline2000/baseline_rough_2000.out
# Stop:    tmux kill-session -t baseline2000
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PY=/home/manas/miniforge3/envs/resloco/bin/python3
LOG_DIR="$ROOT_DIR/log_baseline2000"
OUT="$LOG_DIR/baseline_rough_2000.out"

# CUDA_VISIBLE_DEVICES uses FASTEST_FIRST ordering, so 0 is the RTX 3070 (sm_86).
# The two GTX 1080s are sm_61 and the installed torch ships no sm_61 kernels.
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl

run_job() {
    local t0 rc
    t0=$(date +%s)
    echo "============================================================"
    echo "  baseline_rough_2000  seed=0  started $(date '+%F %T')"
    echo "============================================================"

    "$PY" scripts/train.py \
        --config "$ROOT_DIR/configs/baseline_rough_2000.json" \
        --id baseline_rough_2000 \
        --log_dir "$LOG_DIR" \
        --seed 0 \
        --vec_env_nums 16 \
        --proc_nums 16 \
        --overwrite
    rc=$?

    echo "  baseline_rough_2000 -> $([ $rc -eq 0 ] && echo DONE || echo "FAILED rc=$rc")"\
         "in $(( ($(date +%s) - t0) / 60 )) min at $(date '+%F %T')"
}

SESSION=baseline2000

if [ "${_BASELINE_CHILD:-0}" = "1" ]; then
    run_job
    exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists -- attach with: tmux attach -t $SESSION"
    exit 1
fi

mkdir -p "$LOG_DIR"
# `tee` keeps the scrollback inside tmux while still persisting the log file.
tmux new-session -d -s "$SESSION" \
    "_BASELINE_CHILD=1 '$0' 2>&1 | tee '$OUT'"
echo "launched in tmux session '$SESSION' -> $OUT"
echo "attach: tmux attach -t $SESSION   (detach: Ctrl-b d)"
echo "stop:   tmux kill-session -t $SESSION"
