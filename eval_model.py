#!/usr/bin/env python3
"""
Score a trained wake-word model against the four gates in tuning.md.

Everything here is measured by streaming - `Model.predict_clip` slides the model
over the clip 80 ms at a time, exactly as live detection does - because that is
what the gates are about. `check_model_alignment.py` answers a different question
(where in the window the model wants the phrase) by placing clips at fixed offsets;
a clip that misses at one offset may well fire at the next one in streaming, so the
two scripts are not interchangeable.

Gates (from tuning.md):

    extend + hey_other false accepts at 0.5    < 2/32
    clean positive detection at 0.5            >= 55/56
    detection with a command immediately after >= 27/30
    median latency from end of speech          < 120 ms

Two details of the method matter enough to state:

* Clips are padded with a low noise floor, never digital silence. Pure zeros are a
  pathological input to the melspectrogram and shift scores, so `predict_clip`'s own
  zero padding is bypassed (`padding=0`).
* "End of speech" is the last sample above 2% of peak amplitude. Latency is the audio
  offset where the score first crosses the threshold, minus that marker.

Negatives are reported PER CATEGORY, never pooled. The corpus from
generate_negatives.py is adversarial by construction - a fifth of it is
phrase-extending - so a pooled false-accept rate means nothing. Category comes from
the filename prefix (`extend_000_af_bella.wav` -> `extend`).

Usage:
    python eval_model.py --model my_custom_model/hey_seeree/hey_seeree.onnx
    python eval_model.py --model M --positives my_real_samples/jay
    python eval_model.py --model M --threshold 0.7 --verbose

Needs onnxruntime and an importable openwakeword, so run it in the trainer container
or with PYTHONPATH pointing at an openWakeWord checkout.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io.wavfile

SR = 16000
FRAME = 1280                # predict_clip's step: 80 ms
NOISE_FLOOR = 30.0          # std dev in 16-bit counts; stands in for room tone
PAD_S = 1.0                 # noise before and after each clip
SPEECH_END_FRAC = 0.02      # "end of speech" = last sample above 2% of peak

# Gates from tuning.md, as rates so they survive a different corpus size.
GATE_FALSE_ACCEPT = 2 / 32
GATE_POSITIVE = 55 / 56
GATE_COMMAND = 27 / 30
GATE_LATENCY_MS = 120

# Categories that carry the signal; `general` is the realistic background rate.
ADVERSARIAL = ("extend", "hey_other")

# generate_negatives.py names files "{category}_{index:03d}_{voice}.wav", and two
# category names contain an underscore themselves (hey_other, other_ww). Splitting
# on the first underscore silently renames those to "hey" and "other", which drops
# hey_other out of the adversarial gate entirely.
CATEGORY_RE = re.compile(r"^(.+?)_\d{3}_")


def read_wav(path):
    sr, data = scipy.io.wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if sr != SR:
        return None
    return data.astype(np.int16)


def speech_end(data):
    """Last sample above 2% of peak amplitude, or None if the clip is silent."""
    a = np.abs(data.astype(np.float64))
    peak = a.max()
    if peak <= 0:
        return None
    above = np.flatnonzero(a > SPEECH_END_FRAC * peak)
    return int(above[-1]) if above.size else None


def with_noise_floor(data, rng, pad_s=PAD_S):
    """Pad with room tone rather than zeros, and report where the clip starts."""
    pad = int(SR * pad_s)
    lead = rng.normal(0, NOISE_FLOOR, pad).astype(np.int16)
    tail = rng.normal(0, NOISE_FLOOR, pad).astype(np.int16)
    return np.concatenate([lead, data, tail]), pad


def stream(model, name, audio):
    """Per-frame scores for one clip, plus the audio offset each frame reflects."""
    model.reset()
    frames = model.predict_clip(audio, padding=0)
    scores = np.array([f[name] for f in frames])
    # Frame k is produced after the model has consumed (k+1)*FRAME samples.
    offsets = (np.arange(len(scores)) + 1) * FRAME
    return scores, offsets


def first_crossing(scores, offsets, threshold):
    hit = np.flatnonzero(scores >= threshold)
    return (offsets[hit[0]] if hit.size else None)


def load_dir(directory, recursive=True):
    globber = Path(directory).rglob if recursive else Path(directory).glob
    out, skipped = [], 0
    for wav in sorted(globber("*.wav")):
        data = read_wav(wav)
        if data is None:
            skipped += 1
            continue
        out.append((wav.name, data))
    return out, skipped


def evaluate_positives(model, name, clips, threshold, rng, verbose):
    """Detection rate and latency from end of speech, on clean positives."""
    detected, latencies, misses = 0, [], []
    for clip_name, data in clips:
        audio, pad = with_noise_floor(data, rng)
        scores, offsets = stream(model, name, audio)
        crossing = first_crossing(scores, offsets, threshold)
        if crossing is None:
            misses.append((clip_name, float(scores.max())))
            continue
        detected += 1
        end = speech_end(data)
        if end is not None:
            latencies.append((crossing - (pad + end)) / SR * 1000)
        if verbose:
            print(f"    {clip_name[:36]:<38}peak {scores.max():.3f}  "
                  f"latency {(crossing - (pad + end)) / SR * 1000:>6.0f}ms")
    return detected, latencies, misses


def evaluate_with_command(model, name, clips, commands, threshold, rng, gap_ms, verbose):
    """Detection when a command follows the phrase, with an optional pause between.

    The gap is the discriminating variable: tuning.md measured 20/30 with the command
    butted straight on and 28/30 with 300 ms of pause, which is the signature of a
    model that learned the phrase is followed by quiet.
    """
    detected, misses = 0, []
    for i, (clip_name, data) in enumerate(clips):
        command = commands[i % len(commands)][1]
        gap = np.zeros(int(SR * gap_ms / 1000), dtype=np.int16) if gap_ms else None
        parts = [data] + ([gap] if gap is not None else []) + [command]
        audio, _ = with_noise_floor(np.concatenate(parts), rng)
        scores, _ = stream(model, name, audio)
        if scores.max() >= threshold:
            detected += 1
        else:
            misses.append((clip_name, float(scores.max())))
    if verbose and misses:
        for clip_name, peak in misses[:8]:
            print(f"    missed {clip_name[:36]:<38}peak {peak:.3f}")
    return detected, misses


def evaluate_negatives(model, name, clips, threshold, rng):
    """Max score per clip, grouped by the category in the filename prefix."""
    by_category = defaultdict(list)
    for clip_name, data in clips:
        audio, _ = with_noise_floor(data, rng)
        scores, _ = stream(model, name, audio)
        match = CATEGORY_RE.match(clip_name)
        category = match.group(1) if match else "(uncategorised)"
        by_category[category].append((clip_name, float(scores.max())))
    return by_category


def verdict(ok):
    return "PASS" if ok else "FAIL"


def main():
    parser = argparse.ArgumentParser(
        description="Score a wake-word model against the tuning.md gates",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Trained .onnx or .tflite model")
    parser.add_argument("--positives", default="my_real_samples",
                        help="Directory of positive clips, searched recursively")
    parser.add_argument("--negatives", default="negatives_tts",
                        help="Directory from generate_negatives.py")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--command-gap-ms", type=float, default=300,
                        help="Pause to test alongside the no-pause case (default: %(default)s)")
    parser.add_argument("--limit", type=int, help="Only use the first N clips of each set")
    parser.add_argument("--verbose", action="store_true", help="Print a row per positive")
    args = parser.parse_args()

    # Imported here so --help works without openwakeword installed.
    from openwakeword.model import Model

    # openWakeWord defaults to tflite; pick from the extension so the gates can be
    # scored on the artifact that actually ships, not only on the ONNX it came from.
    framework = "tflite" if Path(args.model).suffix == ".tflite" else "onnx"
    name = Path(args.model).stem
    model = Model(wakeword_models=[args.model], inference_framework=framework)
    rng = np.random.default_rng(0)

    positives, skipped_p = load_dir(args.positives)
    negatives, skipped_n = load_dir(args.negatives)
    if args.limit:
        positives, negatives = positives[:args.limit], negatives[:args.limit]
    if not positives:
        print(f"No usable WAV files in {args.positives}")
        sys.exit(1)
    for label, skipped in (("positives", skipped_p), ("negatives", skipped_n)):
        if skipped:
            print(f"WARNING: skipped {skipped} {label} not at {SR} Hz")

    print("=" * 70)
    print(f"{Path(args.model).name}   threshold {args.threshold}")
    print(f"{len(positives)} positives from {args.positives}, "
          f"{len(negatives)} negatives from {args.negatives}")
    print("=" * 70)

    # --- negatives, per category -------------------------------------------------
    by_category = evaluate_negatives(model, name, negatives, args.threshold, rng)
    print("\nFALSE ACCEPTS BY CATEGORY (never read these pooled)")
    print(f"  {'category':<12}{'n':>4}{'fired':>7}{'rate':>8}{'median':>9}{'worst':>8}")
    adversarial_n = adversarial_fired = 0
    for category in sorted(by_category):
        rows = by_category[category]
        peaks = np.array([p for _, p in rows])
        fired = int((peaks >= args.threshold).sum())
        flag = " <-" if category in ADVERSARIAL else ""
        print(f"  {category:<12}{len(rows):>4}{fired:>7}{fired / len(rows):>7.0%}"
              f"{np.median(peaks):>9.3f}{peaks.max():>8.3f}{flag}")
        if category in ADVERSARIAL:
            adversarial_n += len(rows)
            adversarial_fired += fired

    worst = sorted((r for c in ADVERSARIAL for r in by_category.get(c, [])),
                   key=lambda r: -r[1])[:5]
    if worst and worst[0][1] >= args.threshold:
        print("\n  worst adversarial clips:")
        for clip_name, peak in worst:
            print(f"    {clip_name[:44]:<46}{peak:.3f}")

    # --- positives ---------------------------------------------------------------
    print("\nPOSITIVES")
    detected, latencies, misses = evaluate_positives(
        model, name, positives, args.threshold, rng, args.verbose)
    print(f"  detected            {detected}/{len(positives)} ({detected / len(positives):.0%})")
    if latencies:
        lat = np.array(latencies)
        print(f"  latency from speech end   median {np.median(lat):>6.0f}ms   "
              f"p90 {np.percentile(lat, 90):>6.0f}ms")
        # A clip cannot legitimately be detected before its speech ends. When it
        # happens the clip almost always holds two utterances - the model fires on
        # the first, while the marker sits at the end of the second. The mean is
        # useless in their presence, hence median and p90 above.
        early = int((lat < 0).sum())
        if early:
            print(f"  NOTE: {early} clip(s) fired before their speech-end marker - "
                  f"likely two utterances in one file.")
            print("        check_alignment.py flags these as over 2x the median length.")
    if misses:
        print(f"  missed {len(misses)}:")
        for clip_name, peak in sorted(misses, key=lambda r: -r[1])[:8]:
            print(f"    {clip_name[:44]:<46}{peak:.3f}")

    # --- positives with a command following ---------------------------------------
    commands = [(n, d) for n, d in negatives if n.startswith("command_")]
    detected_cmd = detected_gap = None
    if commands:
        print(f"\nPOSITIVES WITH A COMMAND FOLLOWING ({len(commands)} commands, cycled)")
        detected_cmd, _ = evaluate_with_command(
            model, name, positives, commands, args.threshold, rng, 0, args.verbose)
        detected_gap, _ = evaluate_with_command(
            model, name, positives, commands, args.threshold, rng,
            args.command_gap_ms, False)
        n = len(positives)
        print(f"  command immediately after {detected_cmd}/{n} ({detected_cmd / n:.0%})")
        print(f"  {args.command_gap_ms:.0f}ms pause, then command  "
              f"{detected_gap}/{n} ({detected_gap / n:.0%})")
        if detected_gap - detected_cmd >= max(2, 0.05 * n):
            print("  The pause recovers detections, which is the signature of a model")
            print("  trained on 'wake word, then quiet' (tuning.md, Priority 3).")
    else:
        print(f"\nNo command_*.wav in {args.negatives}; skipping the command-following gate.")

    # --- gates -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GATES")
    rate = adversarial_fired / adversarial_n if adversarial_n else 0.0
    checks = [(f"extend+hey_other false accepts  {adversarial_fired}/{adversarial_n} "
               f"({rate:.0%})", f"< {GATE_FALSE_ACCEPT:.0%}",
               adversarial_n > 0 and rate < GATE_FALSE_ACCEPT)]
    checks.append((f"clean positive detection        {detected}/{len(positives)} "
                   f"({detected / len(positives):.0%})", f">= {GATE_POSITIVE:.0%}",
                   detected / len(positives) >= GATE_POSITIVE))
    if detected_cmd is not None:
        checks.append((f"detection with command after    {detected_cmd}/{len(positives)} "
                       f"({detected_cmd / len(positives):.0%})", f">= {GATE_COMMAND:.0%}",
                       detected_cmd / len(positives) >= GATE_COMMAND))
    if latencies:
        median_latency = float(np.median(latencies))
        checks.append((f"median latency                  {median_latency:.0f}ms",
                       f"< {GATE_LATENCY_MS}ms", median_latency < GATE_LATENCY_MS))
    for text, gate, ok in checks:
        print(f"  [{verdict(ok)}]  {text:<48}{gate}")
    print("=" * 70)


if __name__ == "__main__":
    main()
