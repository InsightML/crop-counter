#!/usr/bin/env bash
# End-to-end CFD-17 benchmark: metadata -> manifest -> capped all-17 subset ->
# fetch -> frozen run -> unfrozen run -> RF-DETR baselines -> one scorer -> MLflow.
#
# Every step is guarded by a real output file, so relaunching the identical
# command after a spot interruption skips the work already on disk and picks the
# run back up from ``last.pt``. Nothing here is interactive.
#
# Usage:
#   examples/FishDetection/scripts/run_all.sh
#   S3_URI=s3://insightml-cfd-benchmark DEVICE=cuda .../run_all.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# --- environment surface ---------------------------------------------------- #
REPO="${REPO:-$(cd "$HERE/../../.." && pwd)}"
PY="${PY:-python}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-$REPO/data/cfd17}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO/weights}"
RUNS_DIR="${RUNS_DIR:-$REPO/runs}"
RESULTS_DIR="${RESULTS_DIR:-$REPO/results}"
CFD_META="${CFD_META:-$REPO/data/cfd/community_fish_detection_dataset.json.zip}"
CFD_META_URL="${CFD_META_URL:-https://lilawildlife.blob.core.windows.net/lila-wildlife/community-fish-detection-dataset/community_fish_detection_dataset.json.zip}"
TRAIN_CAP="${TRAIN_CAP:-20000}"
VAL_CAP="${VAL_CAP:-4000}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-8}"
FETCH_WORKERS="${FETCH_WORKERS:-32}"
MIRROR="${MIRROR:-gcs}"
S3_URI="${S3_URI:-}"
SYNC_EVERY="${SYNC_EVERY:-600}"
RFDETR_NANO_URL="${RFDETR_NANO_URL:-https://github.com/filippovarini/community-fish-detector/releases/download/2026.07.06-release/cfd-rf-detr-nano-640-2026.02.02.cp-011.20260706-release.pth}"
RFDETR_MEDIUM_URL="${RFDETR_MEDIUM_URL:-https://github.com/filippovarini/community-fish-detector/releases/download/2026.07.06-release/cfd-rf-detr-medium-1024-2026.03.24.cp-011.20260706-release.pth}"
ALLOW_LARGE_FETCH="${ALLOW_LARGE_FETCH:-0}"
MAX_FETCH_IMAGES="${MAX_FETCH_IMAGES:-250000}"
CONFIG_FROZEN="${CONFIG_FROZEN:-$REPO/examples/FishDetection/config_cfd17_frozen_8ep.json}"
CONFIG_UNFROZEN="${CONFIG_UNFROZEN:-$REPO/examples/FishDetection/config_cfd17_unfrozen_8ep.json}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-}"
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-crop-counter_CFD17}"

FROZEN_RUN="${FROZEN_RUN:-cfd17_frozen_s0}"
UNFROZEN_RUN="${UNFROZEN_RUN:-cfd17_unfrozen_s0}"

mkdir -p "$RUNS_DIR" "$RESULTS_DIR" "$DATA_ROOT" "$(dirname "$CFD_META")"

# --- 1. logging + the background S3 sync ------------------------------------ #
LOG="$RUNS_DIR/run_all.log"
exec > >(tee -a "$LOG") 2>&1

START_EPOCH=$SECONDS
SYNC_PID=""

sync_once() {
    [ -n "$S3_URI" ] || return 0
    # Checkpoints are the point of the sync: a spot reclaim mid-run is recovered
    # from last.pt, so .pt files are INCLUDED, not excluded.
    aws s3 sync "$RUNS_DIR" "$S3_URI/runs" --only-show-errors || true
    aws s3 sync "$RESULTS_DIR" "$S3_URI/results" --only-show-errors || true
}

start_sync_loop() {
    [ -n "$S3_URI" ] || { echo "[sync] S3_URI empty — no background sync"; return 0; }
    while true; do
        sleep "$SYNC_EVERY"
        sync_once
    done &
    SYNC_PID=$!
    echo "[sync] background sync to $S3_URI every ${SYNC_EVERY}s (pid $SYNC_PID)"
}

on_exit() {
    local rc=$?
    if [ -n "$SYNC_PID" ]; then
        kill "$SYNC_PID" 2>/dev/null || true
        wait "$SYNC_PID" 2>/dev/null || true
    fi
    sync_once
    echo "end   $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "ALL DONE rc=$rc wall=$(( SECONDS - START_EPOCH ))s ($(( (SECONDS - START_EPOCH) / 60 ))m) runs=$RUNS_DIR results=$RESULTS_DIR"
}
trap on_exit EXIT

echo "======================================================================"
echo "start $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "repo=$REPO device=$DEVICE data=$DATA_ROOT runs=$RUNS_DIR results=$RESULTS_DIR"
echo "caps train=$TRAIN_CAP val=$VAL_CAP seed=$SEED epochs=$EPOCHS mirror=$MIRROR"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
else
    echo "gpu: nvidia-smi not present"
fi
echo "======================================================================"

start_sync_loop

# --- 2. CFD metadata -------------------------------------------------------- #
if [ -s "$CFD_META" ]; then
    echo "[skip] metadata already at $CFD_META"
else
    echo "[run ] downloading CFD metadata"
    if command -v wget >/dev/null 2>&1; then
        wget -c -O "$CFD_META" "$CFD_META_URL"
    else
        curl -fL -C - -o "$CFD_META" "$CFD_META_URL"
    fi
fi

# --- 3. per-source manifest ------------------------------------------------- #
MANIFEST_DIR="$DATA_ROOT/manifest"
if [ -f "$MANIFEST_DIR/manifest.csv" ]; then
    echo "[skip] manifest already at $MANIFEST_DIR/manifest.csv"
else
    echo "[run ] manifest"
    "$PY" -m cropcounter.cfd manifest \
        --metadata "$CFD_META" --out "$MANIFEST_DIR" --seed "$SEED" --no-progress
fi

# --- 4. capped all-17 subset ------------------------------------------------ #
if [ -f "$DATA_ROOT/subset_summary.json" ] && [ -f "$DATA_ROOT/val/annotations.json" ]; then
    echo "[skip] subset already at $DATA_ROOT"
else
    echo "[run ] subset (all sources, train<=$TRAIN_CAP val<=$VAL_CAP per source)"
    "$PY" -m cropcounter.cfd subset \
        --metadata "$CFD_META" --out "$DATA_ROOT" --sources all \
        --train-cap "$TRAIN_CAP" --val-cap "$VAL_CAP" --seed "$SEED" --no-progress
fi

if [ -n "$S3_URI" ]; then
    # The COCO jsons alone, never the pixels: a --resume launch restores these and
    # skips the multi-hour streaming pass over the 1.9M-record master file.
    aws s3 sync "$DATA_ROOT" "$S3_URI/data-manifest" \
        --exclude '*/images/*' --exclude '.fetched' --only-show-errors || true
fi

# ``n_download`` is the number of image files ``fetch`` will pull (train + val).
N_DOWNLOAD="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["n_download"])' \
    "$DATA_ROOT/subset_summary.json")"
echo "[info] subset will fetch $N_DOWNLOAD images"
if [ "$N_DOWNLOAD" -gt "$MAX_FETCH_IMAGES" ] && [ "$ALLOW_LARGE_FETCH" != "1" ]; then
    echo "ABORT: $N_DOWNLOAD images > $MAX_FETCH_IMAGES. Expected ~199k for the" >&2
    echo "       capped all-17 subset. Re-run with ALLOW_LARGE_FETCH=1 to proceed." >&2
    exit 1
fi

# --- 5. fetch the pixels ---------------------------------------------------- #
if [ -f "$DATA_ROOT/.fetched" ]; then
    echo "[skip] pixels already fetched ($(cat "$DATA_ROOT/.fetched"))"
else
    echo "[run ] fetch (--max-side 1024 --workers $FETCH_WORKERS --mirror $MIRROR)"
    # A non-zero "failed" count is NOT fatal and must not block the marker: a few
    # source files are 404 on every mirror (the coralscapes PNGs), and fetch drops
    # those images from the live annotations.json — annotations.native.json keeps
    # them — so the split describes exactly what is on disk and needs no pruning
    # step here. Only a real error makes the command exit non-zero.
    "$PY" -m cropcounter.cfd fetch \
        --subset "$DATA_ROOT" --max-side 1024 --workers "$FETCH_WORKERS" \
        --mirror "$MIRROR" --no-progress
    date -u '+%Y-%m-%dT%H:%M:%SZ' > "$DATA_ROOT/.fetched"
fi

# --- 6. the two training runs ----------------------------------------------- #
run_train() {
    local name="$1" config="$2"
    local run_dir="$RUNS_DIR/$name"
    if [ -f "$run_dir/.done" ]; then
        echo "[skip] train $name (.done present)"
        return 0
    fi
    local args=(-m cropcounter.train
        --config "$config"
        --data-root "$DATA_ROOT"
        --weights-dir "$WEIGHTS_DIR"
        --out-dir "$RUNS_DIR"
        --run-name "$name"
        --device "$DEVICE"
        --epochs "$EPOCHS")
    if [ -f "$run_dir/last.pt" ]; then
        echo "[run ] train $name — RESUMING from $run_dir/last.pt"
        args+=(--resume "$run_dir/last.pt")
    else
        echo "[run ] train $name — fresh"
    fi
    "$PY" "${args[@]}"
    touch "$run_dir/.done"
    sync_once
}

run_train "$FROZEN_RUN" "$CONFIG_FROZEN"
run_train "$UNFROZEN_RUN" "$CONFIG_UNFROZEN"

# --- 7. released RF-DETR baselines ------------------------------------------ #
run_baseline() {
    local name="$1" url="$2" side="$3"
    local out="$RESULTS_DIR/$name/predictions.json"
    if [ -f "$out" ]; then
        echo "[skip] baseline $name ($out present)"
        return 0
    fi
    echo "[run ] baseline $name"
    "$PY" "$HERE/rfdetr_baseline.py" \
        --checkpoint-url "$url" \
        --weights-dir "$WEIGHTS_DIR/baselines" \
        --images "$DATA_ROOT/val/images" \
        --annotations "$DATA_ROOT/val/annotations.json" \
        --out "$out" \
        --metrics "$RESULTS_DIR/baseline_metrics.json" \
        --name "$name" \
        --threshold 0.001 \
        --device "$DEVICE" \
        --side "$side"
}

run_baseline rfdetr_nano_640 "$RFDETR_NANO_URL" 640
run_baseline rfdetr_medium_1024 "$RFDETR_MEDIUM_URL" 1024

# --- 8. one scorer over everything ------------------------------------------ #
echo "[run ] evaluate_cfd17"
"$PY" "$HERE/evaluate_cfd17.py" \
    --data-root "$DATA_ROOT" \
    --runs-dir "$RUNS_DIR" \
    --baselines-dir "$RESULTS_DIR" \
    --out "$RESULTS_DIR"

# --- 9. MLflow -------------------------------------------------------------- #
if [ -n "$MLFLOW_TRACKING_URI" ]; then
    for name in "$FROZEN_RUN" "$UNFROZEN_RUN"; do
        [ -d "$RUNS_DIR/$name" ] || continue
        echo "[run ] log_mlflow $name"
        # MLflow is telemetry, never the gate: a tracking outage must not fail the run.
        "$PY" "$HERE/log_mlflow.py" \
            --run-dir "$RUNS_DIR/$name" \
            --results-summary "$RESULTS_DIR/results_summary.json" \
            --experiment "$MLFLOW_EXPERIMENT" \
            --tracking-uri "$MLFLOW_TRACKING_URI" || echo "[warn] MLflow logging failed for $name"
    done
else
    echo "[skip] MLflow (MLFLOW_TRACKING_URI empty)"
fi

# --- 10. the EXIT trap does the final sync and prints ALL DONE -------------- #
