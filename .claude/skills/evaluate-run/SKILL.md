---
name: evaluate-run
description: Evaluate a newly trained wake-word model (.onnx or .tflite) from this repo — held-out detection, false accepts per category, matched-precision comparison against previous runs, and window alignment. Use when a training run finishes, when asked to eval/compare/score a model, or when deciding which model to ship or what detection threshold to deploy at.
---

# Evaluating a training run

Three rules, each learned by getting it wrong across eleven runs. They are in
`tuning.md` in full; the short version is here because ignoring any one of them
produces a confident wrong answer.

**1. Score on recordings made AFTER the model trained.** `train.py` trains on
everything under `my_real_samples/`, so pointing an evaluation there reports training
accuracy. It overstated detection by ~10 points and hid a much larger gap on run-on
speech. Held-out sets live in `my_real_samples_holdout/`.

**2. Never compare models at a fixed threshold.** Two runs of an identical
configuration measured 77% and 67% on held-out run-on speech at threshold 0.5, and
both reached 95% at 8/32 false accepts. What varies between runs is largely where the
score distribution sits, not how well the model separates classes. Always compare at
matched false-accept rates — that is what `compare_models.py` does.

**3. Synthetic evaluation is a lower bound on difficulty, not a gate.** A model
scoring 100% on synthetic "wake word + command" clips detected 46% of real ones. The
synthetic speed sweep, at six voices per point, measures which voices are hard rather
than which speeds.

## Running it

All three tools need `openwakeword` importable and onnxruntime (plus `ai-edge-litert`
for `.tflite`). Either run inside the trainer container, or set
`PYTHONPATH=../openWakeWord` if a checkout is beside this repo.

```bash
# 1. The main comparison: new model against the previous best, matched precision
python compare_models.py --models my_custom_model/hey_seeree/hey_seeree_<new>.onnx \
                                  my_custom_model/hey_seeree/hey_seeree_<prev>.onnx

# 2. Per-category detail for one model: which negatives fire, latency, misses
python eval_model.py --model <model> \
    --positives my_real_samples_holdout/jay --negatives negatives_tts

# 3. Where in the detection window the model expects the phrase.
#    The peak is also the latency floor. Compare against the TRAINED-set clips,
#    not the held-out ones, so it is comparable across runs.
python check_model_alignment.py --model <model> --positives my_real_samples/jay

# 4. Choosing a deployment threshold
python compare_models.py --models <model> --sweep
```

## Reading the results

**Detection**, on held-out clips at matched false accepts. Current best (both 50k-step
runs, at 8/32 adversarial false accepts): **99% plain, 95% run-on**. Run-on is the
harder and more important number — it is the commonest real usage, and it was 5%
before run-on positives existed.

**False accepts**, read PER CATEGORY, never pooled. The corpus is adversarial by
construction — a fifth of it is phrase-extending — so a pooled rate is meaningless.

- `extend`, `hey_other` — phonetic near-misses ("hey serious"). The unsolved problem:
  6–8/32 across every run. This is what the matched comparison is keyed on.
- `general`, `command`, `other_ww`, `running` — ordinary speech. Should be ~0. Any
  movement here is more serious than movement in `extend`.

**Alignment**, from `check_model_alignment.py`. Peak should be ~160 ms with the band
starting near 80 ms. A peak near 400+ ms means the training clips carried trailing
silence; a band reaching 0 ms means the model fires without hearing the word end,
which tripled false accepts when it happened in run 5.

**Latency** should be under ~120 ms. A *negative* median means the model fires before
the speech-end marker — normal for this corpus, since the marker sits on room tone.

## Judging a result

A difference of a few points is not meaningful. Two runs of the same configuration
have measured 10 points apart at a fixed threshold. Only trust an effect that is
large (10+ points on run-on at matched precision) or replicated.

If detection and false accepts both move the same way, that is an operating-point
shift, not a better model — check the matched comparison before believing it. This
has happened twice: `max_negative_weight` 4000, and `--training-steps` 100k.

## After evaluating

Record the result in `tuning.md` as a new section at the top: the commit hash, what
changed against the previous run, what was predicted, and what the measurement said —
including predictions that turned out wrong. Several conclusions in that file were
reversed by later runs, and the reasoning is worth more than the conclusion.

If the model is being shipped, convert it and verify the conversion, since a
wrong-axis tflite loads cleanly and detects nothing:

```bash
python onnx2tflite.py <model>.onnx -o <model>.tflite   # checks against the ONNX
python compare_models.py --models <model>.tflite       # should match the .onnx
```
