#!/usr/bin/env python3
"""Emit the ESPHome manifest for a trained microWakeWord model.

microWakeWord produces only the .tflite. ESPHome needs a JSON manifest beside it,
and the values in it are not cosmetic - two of them change detection behaviour and
one has to match how the model was trained.

    python -m mww.manifest --wake-word "hey seeree" --run 20260902-063606

THE CUTOFF COMES FROM THE MEASUREMENT, NOT FROM A DEFAULT. Training writes
tflite_streaming_roc.txt: false rejection rate and false accepts per hour at every
cutoff. Picking a number without reading it is the same mistake as evaluating an
openWakeWord model at threshold 0.5, which tuning.md opens by warning about - two
runs of one configuration read 77% and 67% there.

    Cutoff 0.86: frr=0.0245; faph=0.000     <- zero false accepts, 97.6% recall
    Cutoff 0.70: frr=0.0168; faph=0.187
    Cutoff 0.60: frr=0.0116; faph=0.281

`--max-faph` picks the lowest cutoff meeting a false-accepts-per-hour budget, which
maximises recall subject to that budget. Default 0.0: the most conservative choice
the measurement supports.

SLIDING_WINDOW_SIZE AND THE CUTOFF ARE A PAIR. The ROC is computed with a sliding
window average of 5 (microwakeword/test.py:301), so a cutoff read off that table is
only valid at `sliding_window_size: 5`. Changing one without re-deriving the other
silently moves the operating point.

FEATURE_STEP_SIZE MUST MATCH TRAINING. It is how often ESPHome's preprocessor emits
a feature vector, and the model was trained on features at `window_step_ms`. A
mismatch feeds the model a different time base than it learned - it will not error,
it will just detect badly. Published models often show 10 because they were trained
that way; this repo trains at 20.

TENSOR_ARENA_SIZE IS A GUESS AND HAS TO BE CHECKED ON DEVICE. It is the working
memory TFLite Micro allocates, and it cannot be computed here - it depends on the
arena planner in the ESPHome build. Too small and ESPHome fails to allocate at boot
with a clear error, so the failure is loud. Start with the default and raise it if
the device complains.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mww import config as mww_config  # noqa: E402

QUANT_DIR = "tflite_stream_state_internal_quant"
MODEL_FILE = "stream_state_internal_quant.tflite"
ROC_FILE = "tflite_streaming_roc.txt"

# microwakeword/test.py:301 - the ROC is computed with this window, so a cutoff
# taken from it is only meaningful alongside the same value here.
SLIDING_WINDOW_SIZE = 5

# Not computable from here; see the module docstring. ESPHome fails loudly at boot
# if it is too small.
DEFAULT_TENSOR_ARENA_SIZE = 30000
MINIMUM_ESPHOME_VERSION = "2024.7.0"

ROC_LINE = re.compile(r"Cutoff\s+([\d.]+)\s*:\s*frr\s*=\s*([\d.]+)\s*;\s*faph\s*=\s*([\d.]+)")


def parse_roc(path: Path):
    """[(cutoff, frr, faph), ...] from tflite_streaming_roc.txt."""
    rows = []
    for line in path.read_text().splitlines():
        m = ROC_LINE.search(line)
        if m:
            rows.append(tuple(float(g) for g in m.groups()))
    return rows


def choose_cutoff(rows, max_faph):
    """Lowest cutoff whose faph is within budget - i.e. best recall for that budget.

    Lower cutoff means more detections, so more false accepts AND fewer false
    rejects. Walking up from the lowest cutoff and taking the first that fits the
    budget gives the most sensitive operating point that still meets it.
    """
    eligible = [r for r in rows if r[2] <= max_faph]
    if not eligible:
        best = min(rows, key=lambda r: r[2])
        raise SystemExit(
            f"no cutoff achieves faph <= {max_faph}. The lowest measured is "
            f"{best[2]} at cutoff {best[0]} (frr {best[1]}). Raise --max-faph, or "
            f"treat this as the model not being good enough to deploy.")
    return min(eligible, key=lambda r: r[0])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wake-word", default="hey seeree")
    p.add_argument("--run", required=True, help="run directory name under the model dir")
    p.add_argument("--models-dir", default="mww_models")
    p.add_argument("--max-faph", type=float, default=0.0,
                   help="false accepts per hour budget (default: %(default)s)")
    p.add_argument("--tensor-arena-size", type=int, default=DEFAULT_TENSOR_ARENA_SIZE)
    p.add_argument("--author", default=None)
    p.add_argument("--website", default=None)
    p.add_argument("--version", type=int, default=2)
    p.add_argument("--languages", default="en")
    args = p.parse_args()

    safe = args.wake_word.replace(" ", "_").lower()
    run_dir = Path(args.models_dir) / safe / args.run / QUANT_DIR
    model = run_dir / MODEL_FILE
    roc = run_dir / ROC_FILE

    for path in (model, roc):
        if not path.is_file():
            sys.exit(f"not found: {path}")
    if model.stat().st_size < 1024:
        sys.exit(f"{model} is {model.stat().st_size} bytes - not a usable model")

    rows = parse_roc(roc)
    if not rows:
        sys.exit(f"no 'Cutoff ...: frr=...; faph=...' lines in {roc}")

    cutoff, frr, faph = choose_cutoff(rows, args.max_faph)
    print(f"{len(rows)} cutoffs measured; {roc}")
    print(f"  chosen cutoff {cutoff}: {100 * (1 - frr):.2f}% recall, "
          f"{faph} false accepts/hour (budget {args.max_faph})")

    manifest = {
        "type": "micro",
        "wake_word": args.wake_word,
        "model": model.name,
        "trained_languages": args.languages.split(","),
        "version": args.version,
        "micro": {
            "probability_cutoff": cutoff,
            # Must equal the window_step_ms the model was TRAINED with.
            "feature_step_size": mww_config.WINDOW_STEP_MS,
            # Must equal the window the ROC above was computed with.
            "sliding_window_size": SLIDING_WINDOW_SIZE,
            "tensor_arena_size": args.tensor_arena_size,
            "minimum_esphome_version": MINIMUM_ESPHOME_VERSION,
        },
    }
    if args.author:
        manifest["author"] = args.author
    if args.website:
        manifest["website"] = args.website

    out = run_dir / f"{safe}.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"\nDeploy both, side by side:\n  {model}\n  {out}")
    print(f"\ntensor_arena_size is {args.tensor_arena_size}, which is a starting "
          f"guess.\nIf ESPHome fails to allocate at boot, raise it and reflash.")


if __name__ == "__main__":
    main()
