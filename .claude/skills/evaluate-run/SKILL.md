---
name: evaluate-run
description: Evaluate a newly trained wake-word model from this repo — openWakeWord (.onnx/.tflite) or microWakeWord (manifest .json) — covering held-out detection per speaker, false accepts per category, matched-precision comparison against previous runs, deployment threshold choice, and window alignment. Use when a training run finishes, when asked to eval/compare/score a model, or when deciding which model to ship or what detection threshold to deploy at.
---

# Evaluating a training run

Four rules, each learned by getting it wrong. They are in `tuning.md` in full; the
short version is here because ignoring any one produces a confident wrong answer.

**1. Score on recordings made AFTER the model trained.** `train.py` trains on
everything under `my_real_samples/`, so pointing an evaluation there reports training
accuracy. It overstated detection by ~10 points and hid a much larger gap on run-on
speech. Held-out sets live in `my_real_samples_holdout/`.

**2. Never compare models at a fixed threshold.** Two runs of an identical
configuration measured 77% and 67% on held-out run-on speech at threshold 0.5, and
both reached 95% at 8/32 false accepts. What varies between runs is largely where the
score distribution sits, not how well the model separates classes. Always compare at
matched false-accept rates — that is what `compare_models.py` does. **This matters even
more across the two trainers, which share no threshold scale at all.**

**3. Score every speaker separately, and never pool them.** Added after a run found
the second speaker — a 4-year-old — at 24% detection while the pooled headline said
99%. A speaker missing from the holdout does not make the number worse, it makes the
number meaningless. `my_real_samples_holdout/<speaker>/` and `<speaker>_runon/`. This
rule paid out again on the first microWakeWord model: level with openWakeWord on jay,
clearly worse on ryan and jen.

**4. Synthetic evaluation is a lower bound on difficulty, not a gate.** A model
scoring 100% on synthetic "wake word + command" clips detected 46% of real ones.

## Running it

**Use the `eval` image. It builds native on the Mac and needs no GPU.** The two
trainer images cannot host evaluation — `trainer` pins numpy<2, `mww` needs numpy>=2
and CUDA TensorFlow, and both are linux/amd64 server images.

```bash
docker compose build eval        # once; ~1 min, no TensorFlow, no emulation
docker compose run --rm eval python compare_models.py ...
```

`eval/backends.py` picks the backend by inspecting the model and **runs the inference
code the deployment target runs** — Linux Voice Assistant's `pymicro-wakeword` and
`pyopen-wakeword`, driven as its `__main__.py` drives them.

| what you pass | backend |
|---|---|
| `<run>/hey_seeree.json` | microWakeWord, manifest included — **prefer this** |
| `stream_state_internal_quant.tflite` | microWakeWord without its manifest |
| `hey_seeree_<commit>.tflite` | openWakeWord on the deployment runtime |
| `hey_seeree_<commit>.onnx` | openWakeWord via `openwakeword.model.Model` — the tuning.md path, **NOT deployment** |

Pass the **manifest `.json`** for a microWakeWord model wherever one exists. It is what
the runtime loads, so `sliding_window_size` and `probability_cutoff` get tested along
with the weights. That is how a manifest shipping `probability_cutoff: 1.0` — at which
the model cannot fire — was caught.

```bash
# 0. Self-check a backend before trusting any number from it.
#    Proves state does not leak between clips, and reports score resolution.
docker compose run --rm eval python -m eval.backends --model <model>

# 1. The main comparison: new model against the previous best, matched precision.
#    Run it ONCE PER SPEAKER — see rule 3.
docker compose run --rm eval python compare_models.py \
    --models <new> <prev> \
    --positives my_real_samples_holdout/jay --runon my_real_samples_holdout/jay_runon

# 2. Per-category detail for one model: which negatives fire, latency, misses, gates
docker compose run --rm eval python eval_model.py --model <model> \
    --positives my_real_samples_holdout/jay --threshold <operating point>

# 3. Choosing a deployment threshold
docker compose run --rm eval python compare_models.py --models <model> --sweep

# 4. openWakeWord .onnx only: where in the window the model expects the phrase.
#    Compare against TRAINED-set clips, not held-out, so it is comparable across runs.
#    This one does NOT go through eval/backends.py — it has its own loader.
docker compose run --rm eval python -m eval.check_model_alignment \
    --model <model>.onnx --positives my_real_samples/jay
```

## Reading the results

**Detection**, on held-out clips at matched false accepts, per speaker. Run-on is the
harder and more important number — it is the commonest real usage, and it was 5%
before run-on positives existed.

**False accepts**, read PER CATEGORY, never pooled. The corpus is adversarial by
construction — a fifth of it is phrase-extending — so a pooled rate is meaningless.

- `extend`, `hey_other` — phonetic near-misses ("hey serious"). The unsolved problem:
  6–8/32 on every openWakeWord run since run 6, and 8/32 on the first microWakeWord
  model too. Not a frontend problem. This is what the matched comparison is keyed on.
- `general`, `command`, `other_ww`, `running` — ordinary speech. Should be ~0. Any
  movement here is more serious than movement in `extend`.

**Latency** should be under ~120 ms. It is measured identically for both backends and
is the one number that transfers directly between them. A *negative* median means the
model fires before the speech-end marker — normal for this corpus, since the marker
sits on room tone.

**Alignment** (openWakeWord `.onnx` only). Peak should be ~160 ms with the band
starting near 80 ms. A peak near 400+ ms means the training clips carried trailing
silence; a band reaching 0 ms means the model fires without hearing the word end,
which tripled false accepts in run 5.

`eval/check_model_alignment.py` reaches for `ai-edge-litert` on a `.tflite`, which the eval
image does not carry — it uses the deployment runtime's bundled interpreter instead —
so the `.tflite` path needs the trainer image. The tool's framing also does not
transfer to microWakeWord — a streaming detector over a 1500 ms window with a
sliding-window average — so there is no alignment measurement for those yet.

### microWakeWord: threshold semantics are different, in two ways

**Its usable range is 0.25 upwards, not 0.01 upwards.** The first model's resting
score is 0.245, so at 0.15 — openWakeWord's shipping threshold — *every* negative in
the corpus fires, 32/32 adversarial and 68/68 ordinary. Carrying an openWakeWord
threshold habit across produces a detector that never stops firing. `compare_models.py`
sweeps up to 0.95 for this reason.

**"Threshold" is two parameters**: `probability_cutoff` and `sliding_window_size`. Fix
the window per comparison (it defaults to the manifest's value) and sweep the cutoff.
Report the window size alongside every number — a cutoff means nothing without it.

Score resolution is ~0.0008 (int8 output, 256 levels, averaged over a window of 5).
Sweeping finer than that measures quantization, not the model.

## Judging a result

A difference of a few points is not meaningful. Two runs of the same configuration
have measured 10 points apart at a fixed threshold. Only trust an effect that is
large (10+ points on run-on at matched precision) or replicated.

If detection and false accepts both move the same way, that is an operating-point
shift, not a better model — check the matched comparison before believing it. This
has happened twice: `max_negative_weight` 4000, and `--training-steps` 100k.

## After evaluating

Record the result as a new section at the top of the right notebook — **`tuning.md` for
openWakeWord, `tuning_mww.md` for microWakeWord**. Keep them separate: different
frontend, different window, different threshold semantics, and pooling them produces
exactly the confident-irrelevant reading rule 3 exists to prevent.

Record the commit or artifact, what changed against the previous run, what was
predicted, and what the measurement said — **including predictions that turned out
wrong**. Several conclusions in those files were reversed by later runs, and the
reasoning is worth more than the conclusion.

**Shipping an openWakeWord model** — convert and verify, since a wrong-axis tflite
loads cleanly and detects nothing:

```bash
docker compose run --rm --no-deps trainer \
    python onnx2tflite.py <model>.onnx -o <model>.tflite
docker compose run --rm eval python compare_models.py --models <model>.onnx <model>.tflite
```

The `.tflite` also moves it onto the deployment runtime — `pyopen-wakeword` is
TFLite-only, so an `.onnx` cannot be measured as it will actually run.

**Shipping a microWakeWord model** — cut the manifest from the measurement, not from a
default:

```bash
# --max-faph 0.0 (the default) will REFUSE: no measured cutoff reaches zero false
# accepts per hour while still detecting anything. Pick a real budget.
docker compose run --rm eval python -m mww.manifest --wake-word "hey seeree" \
    --models-dir my_custom_model --run mww --max-faph <budget>

docker compose run --rm eval python eval_model.py --model <run>/<wake_word>.json \
    --threshold <the cutoff it chose>
```

`mww/manifest.py` reads `tflite_streaming_roc.txt`, which is scored on ambient audio
and contains none of this repo's adversarial negatives — so it cannot see the `extend`
problem. **Always confirm the chosen cutoff against the holdout with `eval_model.py`
before deploying it.**
