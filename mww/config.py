"""Emit a microWakeWord training YAML pointing at this repo's corpus.

WHY A GENERATOR AND NOT A CHECKED-IN YAML. The same reason train.py builds
openWakeWord's config from a template rather than shipping one: the paths, the
per-set weights and the clip counts all move together, and a hand-edited YAML drifts
from what the corpus actually contains. This mirrors create_config in train.py.

THE KEY THING READING THE SOURCE CHANGED. A feature set can be `type: clips` as well
as `type: mmap` (microwakeword/data.py:405-452). `Clips(input_directory, file_pattern)`
reads a directory of audio files, so `corpus/` output feeds the trainer directly and
spectrograms are generated on the fly. Only the pre-generated ambient negatives from
Hugging Face arrive as RaggedMmap. The plan originally scoped a conversion step for
our own clips; there is none to build.

DERIVED KEYS ARE NOT WRITTEN HERE. `spectrogram_length`,
`spectrogram_length_final_layer`, `training_input_shape` and `stride` are computed by
model_train_eval.py:60-93 from `clip_duration_ms`, `window_step_ms` and the model
flags. Writing them here would be duplicating a calculation that the trainer will
redo, and a stale copy is worse than none.

    python -m mww.config --wake-word "hey seeree" --out training_parameters.yaml
"""

import argparse
from pathlib import Path

import yaml

# 1500 ms, against openWakeWord's 2000 ms window. This is the first place the two
# pipelines genuinely diverge in a way that affects the corpus rather than the code:
# a clip that sits comfortably in a 2 s window may not in 1.5 s. corpus/augment.py's
# trimming makes this survivable - it is why the phrase is flush to the end - but the
# alignment reasoning in tuning.md does NOT carry over unchecked.
CLIP_DURATION_MS = 1500

# 10 ms in the preprocessor, 20 ms as the model's window step. Upstream's default.
WINDOW_STEP_MS = 20

# Augmentation corpora already on disk from setup-data.sh. mWW's Augmentation takes
# the same two things openWakeWord's does, from the same downloads.
IMPULSE_DIRS = ["data/mit_rirs"]
BACKGROUND_DIRS = ["data/audioset_16k", "data/fma"]

# Upstream notebook defaults, kept verbatim as the starting point. Change one at a
# time and record it, tuning.md style - the notebook's own README says a usable model
# takes a lot of experimentation, which is seventeen runs of this repo restated.
DEFAULT_TRAINING_STEPS = [10000]
DEFAULT_LEARNING_RATES = [0.001]
DEFAULT_BATCH_SIZE = 128
DEFAULT_EVAL_STEP_INTERVAL = 500
DEFAULT_POSITIVE_CLASS_WEIGHT = [1]
DEFAULT_NEGATIVE_CLASS_WEIGHT = [20]


def clips_feature_set(directory, truth, sampling_weight, penalty_weight,
                      impulse_dirs, background_dirs, truncation_strategy="default",
                      slide_frames=10, step_ms=WINDOW_STEP_MS):
    """A feature set generated on the fly from a directory of WAVs.

    `slide_frames` is 10 for training and validation and 1 for testing upstream: >1
    yields several overlapping spectrograms per clip by dropping end frames, which
    simulates the sequential inputs a streaming model actually sees. Testing wants the
    real thing, so it uses 1.
    """
    return {
        "type": "clips",
        "truth": truth,
        "sampling_weight": sampling_weight,
        "penalty_weight": penalty_weight,
        "truncation_strategy": truncation_strategy,
        "clips_settings": {
            "input_directory": str(directory),
            "file_pattern": "*.wav",
            # No remove_silence here: corpus/augment.py has already trimmed, with a
            # method calibrated on this corpus. Letting webrtcvad trim a second time
            # would stack two different silence definitions on the same clips.
            "remove_silence": False,
            "random_split_seed": 10,
            "split_count": 0.1,
        },
        "augmentation_settings": {
            "augmentation_duration_s": CLIP_DURATION_MS / 1000.0,
            "impulse_paths": [str(p) for p in impulse_dirs],
            "background_paths": [str(p) for p in background_dirs],
            "background_min_snr_db": -5,
            "background_max_snr_db": 10,
            "min_jitter_s": 0.195,
            "max_jitter_s": 0.205,
        },
        "spectrogram_generation_settings": {
            "step_ms": step_ms,
            "slide_frames": slide_frames,
        },
    }


def mmap_feature_set(features_dir, truth, sampling_weight, penalty_weight,
                     truncation_strategy="truncate_start"):
    """A pre-generated RaggedMmap set - the Hugging Face ambient negatives.

    Layout is <features_dir>/{training,validation,testing,testing_ambient,
    validation_ambient}/**/*_mmap/ (data.py:170-190).
    """
    return {
        "type": "mmap",
        "features_dir": str(features_dir),
        "truth": truth,
        "sampling_weight": sampling_weight,
        "penalty_weight": penalty_weight,
        "truncation_strategy": truncation_strategy,
    }


def build(wake_word, positives_dir, negatives_dir, ambient_dirs, output_dir,
          data_dir=".", training_steps=None, learning_rates=None,
          batch_size=DEFAULT_BATCH_SIZE, negative_class_weight=None):
    safe = wake_word.replace(" ", "_").lower()
    data = Path(data_dir)
    impulse = [data / Path(p).name for p in IMPULSE_DIRS]
    background = [data / Path(p).name for p in BACKGROUND_DIRS]

    features = [
        # Positives from mww/corpus.py: synthetic voices plus real recordings,
        # trimmed and carrying the child-range copies, in mWW's own directory.
        #
        # Real recordings are NOT duplicated here the way --real-copies duplicates
        # them for openWakeWord. That trick exists because openWakeWord augments by
        # globbing the directory once, so N copies become N augmented variants. mWW
        # augments on every read, so copies would only bias sampling - and
        # `sampling_weight` below is the honest knob for that. See corpus/real.py.
        clips_feature_set(positives_dir, truth=True, sampling_weight=2.0,
                          penalty_weight=1.0, impulse_dirs=impulse,
                          background_dirs=background),
        # This repo's ADVERSARIAL negatives - "hey serious", "hey Sienna". ~100 clips
        # against ambient sets orders of magnitude larger, so they need a sampling
        # weight that keeps them visible. This is the per-set lever openWakeWord did
        # not have: there, max_negative_weight applied to the whole negative class.
        clips_feature_set(negatives_dir, truth=False, sampling_weight=2.0,
                          penalty_weight=1.0, impulse_dirs=impulse,
                          background_dirs=background),
    ]
    for d in ambient_dirs:
        features.append(mmap_feature_set(d, truth=False, sampling_weight=1.0,
                                         penalty_weight=1.0))

    return {
        "window_step_ms": WINDOW_STEP_MS,
        "clip_duration_ms": CLIP_DURATION_MS,
        "batch_size": batch_size,
        "training_steps": training_steps or DEFAULT_TRAINING_STEPS,
        "learning_rates": learning_rates or DEFAULT_LEARNING_RATES,
        "positive_class_weight": DEFAULT_POSITIVE_CLASS_WEIGHT,
        "negative_class_weight": negative_class_weight or DEFAULT_NEGATIVE_CLASS_WEIGHT,
        "eval_step_interval": DEFAULT_EVAL_STEP_INTERVAL,
        # Upstream's two-step selection: get false accepts per hour under target
        # first, then maximise recall. The same shape as this repo's rule that
        # detection is only comparable at matched false accepts.
        "minimization_metric": None,
        "target_minimization": 0.5,
        "maximization_metric": "average_viable_recall",
        "train_dir": str(Path(output_dir) / safe),
        "summaries_dir": str(Path(output_dir) / safe / "summaries"),
        "features": features,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wake-word", default="hey seeree")
    # mWW owns its corpus, built by mww/corpus.py. Reading the openWakeWord one in
    # place was tried and rejected: train.py rmtree's it at the start of every run.
    p.add_argument("--positives", default=None,
                   help="default: <corpus-root>/<wake_word>/mww/positives")
    p.add_argument("--negatives", default=None,
                   help="default: <corpus-root>/<wake_word>/mww/negatives")
    p.add_argument("--corpus-root", default="my_custom_model",
                   help="corpora live at <root>/<wake_word>/{oww,mww}/ "
                        "(default: %(default)s)")
    p.add_argument("--ambient", nargs="*", default=[],
                   help="RaggedMmap feature dirs for the ambient negatives")
    p.add_argument("--data-dir", default=".",
                   help="where mit_rirs/audioset_16k/fma live (default: %(default)s)")
    p.add_argument("--output-dir", default="mww_models")
    p.add_argument("--training-steps", type=int, nargs="+")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--out", default="training_parameters.yaml")
    args = p.parse_args()

    safe = args.wake_word.replace(" ", "_").lower()
    corpus = Path(args.corpus_root) / safe / "mww"
    positives = Path(args.positives) if args.positives else corpus / "positives"
    negatives = Path(args.negatives) if args.negatives else corpus / "negatives"

    for label, d in (("positives", positives), ("negatives", negatives)):
        n = len(list(d.glob("*.wav"))) if d.is_dir() else 0
        print(f"  {label:<10} {d}  ({n} wav)")
        if n == 0:
            print("           EMPTY OR MISSING. Build it first:")
            print(f'             python -m mww.corpus --wake-word "{args.wake_word}"')

    cfg = build(args.wake_word, positives, negatives, args.ambient,
                args.output_dir, data_dir=args.data_dir,
                training_steps=args.training_steps, batch_size=args.batch_size)
    Path(args.out).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {args.out}: {len(cfg['features'])} feature sets, "
          f"clip_duration_ms={cfg['clip_duration_ms']}")
    if not args.ambient:
        print("  WARNING: no ambient negative sets. Training against this repo's ~100")
        print("           adversarial negatives alone will produce a model that fires")
        print("           on ordinary speech - those sets are what teach silence and")
        print("           background. Fetch them first (plan.md phase 2).")


if __name__ == "__main__":
    main()
