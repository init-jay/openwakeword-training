#!/usr/bin/env bash
#
# Run a full training pass on the trainer host.
#
# Wraps the sequence that has to happen in order, including the two steps that are
# easy to forget and expensive to get wrong:
#
#   * Kokoro must be UP for generation and DOWN for training. The GPU-resident
#     feature patch holds ~16.6 GiB of VRAM, and openwakeword's validation step
#     allocates ~2.76 GiB in one go. With Kokoro's two CUDA contexts (~2.4 GiB) still
#     resident, that overflows - which killed a run at 37,500 of 50,000 steps, after
#     generation and feature computation had already completed.
#
#   * The model must be checked for freshness. train.py now verifies this itself,
#     but the check is repeated here against the file you are about to copy off the
#     box, because a stale model was evaluated twice before identical checksums gave
#     it away.
#
# Usage:
#   ./run-training.sh "hey seeree"
#   ./run-training.sh "hey seeree" --samples-per-voice 400 --training-steps 100000
#
# Any extra arguments are passed through to train.py.

set -euo pipefail

WAKE_WORD="${1:-}"
if [[ -z "$WAKE_WORD" ]]; then
    echo "usage: $0 \"wake word\" [extra train.py args...]" >&2
    exit 2
fi
shift

cd "$(dirname "$0")"

# tr rather than ${x,,} so this does not need bash 4 (macOS ships 3.2).
SAFE_NAME="$(printf '%s' "$WAKE_WORD" | tr ' [:upper:]' '_[:lower:]')"
MODEL="my_custom_model/${SAFE_NAME}.onnx"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="training-${SAFE_NAME}-${STAMP}.log"

# Record the current model so a stale one cannot be mistaken for this run's output.
BEFORE_SUM=""
[[ -f "$MODEL" ]] && BEFORE_SUM="$(md5sum "$MODEL" | cut -d' ' -f1)"

cleanup() {
    # Always leave Kokoro running: the next run needs it, and a stopped container
    # produces a confusing "no usable Kokoro servers" failure at startup.
    docker compose start kokoro kokoro2 >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "=== $(date '+%H:%M:%S')  starting Kokoro"
docker compose up -d kokoro kokoro2

# Wait for readiness rather than assuming: the GPU image spends a while loading
# voices, and train.py's probe would otherwise fail on a container that is up but
# not yet serving.
for name in kokoro:8880 kokoro2:8881; do
    port="${name##*:}"
    for _ in $(seq 1 60); do
        curl -sf "http://localhost:${port}/v1/audio/voices" >/dev/null 2>&1 && break
        sleep 2
    done
done
echo "=== $(date '+%H:%M:%S')  Kokoro ready"

# Generation and feature computation. Kokoro is needed for the first, and the GPU
# headroom it occupies is harmless until training starts.
echo "=== $(date '+%H:%M:%S')  training (log: $LOG)"
: > "$LOG"

# Stop Kokoro the moment feature computation finishes, freeing its ~2.4 GiB before
# openwakeword's validation allocation needs it. Started before training so the
# marker cannot be missed.
( tail -f "$LOG" 2>/dev/null | grep -q "Training model" \
    && echo "=== $(date '+%H:%M:%S')  training stage reached, stopping Kokoro" \
    && docker compose stop kokoro kokoro2 >/dev/null 2>&1 ) &
WATCH_PID=$!

# Foreground, so PIPESTATUS gives docker's exit code rather than tee's - a
# backgrounded pipeline would report tee's status and call a failed run a success.
set +e
docker compose run --rm trainer \
    python train.py --wake-word "$WAKE_WORD" --data-dir /app/data "$@" 2>&1 \
    | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

# `wait` after `kill` suppresses bash's asynchronous "Terminated" job-control
# message, which would otherwise print in the middle of the summary.
{ kill "$WATCH_PID" 2>/dev/null; pkill -P "$WATCH_PID" 2>/dev/null; \
  wait "$WATCH_PID"; } 2>/dev/null || true

echo
if [[ $STATUS -ne 0 ]]; then
    echo "=== TRAINING FAILED (exit $STATUS) - see $LOG"
    exit "$STATUS"
fi

if [[ ! -f "$MODEL" ]]; then
    echo "=== TRAINING FAILED - $MODEL does not exist"
    exit 1
fi

AFTER_SUM="$(md5sum "$MODEL" | cut -d' ' -f1)"
if [[ -n "$BEFORE_SUM" && "$BEFORE_SUM" == "$AFTER_SUM" ]]; then
    echo "=== TRAINING FAILED - $MODEL is unchanged from before this run."
    echo "    It is the PREVIOUS model. Do not evaluate or deploy it."
    exit 1
fi

# Name the output by commit so a model can be traced back to the code that made it.
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo "$STAMP")"
DIRTY=""
git diff --quiet 2>/dev/null || DIRTY="-dirty"
TAGGED="my_custom_model/${SAFE_NAME}/${SAFE_NAME}_${COMMIT}${DIRTY}.onnx"
mkdir -p "$(dirname "$TAGGED")"
cp "$MODEL" "$TAGGED"

echo "=== $(date '+%H:%M:%S')  DONE"
echo "    $TAGGED  ($(du -h "$TAGGED" | cut -f1), md5 ${AFTER_SUM:0:8})"
[[ -n "$DIRTY" ]] && echo "    NOTE: working tree was dirty - this model is not reproducible from $COMMIT"
echo
echo "    scp to the eval machine, then:"
echo "      python compare_models.py --models <new> <previous-best>"
