# Tuning notebook for microWakeWord

**This is a lab notebook, not reference documentation.** One section per training run,
newest first, each recording the hypothesis, the artifact it ran on, and what the
measurement said - including the predictions that turned out wrong.

Separate from `tuning.md` on purpose (plan.md, "What this plan does not do"): different
frontend, different window, different threshold semantics. Pooling the two would
produce exactly the confident-irrelevant reading the per-speaker rule exists to
prevent. **The four rules at the top of `tuning.md` all still apply**, and rule 2 -
compare at matched false accepts, never at a fixed threshold - applies harder here,
because the two backends do not share a threshold scale at all.

## How these numbers are measured

`eval/backends.py` runs **the inference code the deployment target runs**. The target
is Linux Voice Assistant (OHF-Voice/linux-voice-assistant), which implements no
inference of its own and depends on `pymicro-wakeword` and `pyopen-wakeword`; both are
driven here exactly as LVA's `__main__.py` drives them, including chunk sizes. The
microWakeWord model is loaded from its **manifest .json**, not the bare `.tflite`, so
`sliding_window_size` is under test alongside the weights.

    docker compose build eval          # native arm64, runs on the Mac
    docker compose run --rm eval python -m eval.compare_models --models A B --sweep

One asymmetry to keep in view, because it is not yet resolved: **the openWakeWord side
is NOT measured on the deployment runtime.** `pyopen-wakeword` is TFLite-only and every
recent openWakeWord ship candidate here is `.onnx`, so `d1bb9f4` is scored through
`openwakeword.model.Model` - the path all seventeen runs of `tuning.md` were measured
on. That makes it comparable with the notebook and *not* a deployment measurement.
Converting `d1bb9f4` with `train/oww/onnx2tflite.py` and re-scoring is the outstanding job.

---

## Run 1: first microWakeWord model - competitive on jay, worse on the other two

Artifact: `my_custom_model/hey_seeree/mww/tflite_stream_state_internal_quant/`,
mixednet at upstream notebook defaults, 10 ms feature step, 1500 ms clip, corpus
Piper-only with no run-on positives (plan.md phase 2, "Still to build").

Compared against `d1bb9f4` (run 17), the openWakeWord ship candidate.

### Detection at matched adversarial false accepts, per speaker, never pooled

plain / run-on. `oww` is `d1bb9f4`, `mww` is this model.

| adv FA | jay oww (35/57) | jay mww | ryan oww (6/14) | ryan mww | jen oww (10) | jen mww |
|---|---|---|---|---|---|---|
| 2/32  | 94/61  | 91/53  | 17/57  | 33/36 | 90  | 20 |
| 4/32  | 100/84 | 97/91  | 50/86  | 50/57 | 100 | 50 |
| 6/32  | 100/86 | 97/93  | 50/86  | 67/64 | 100 | 60 |
| 8/32  | 100/95 | 100/95 | 100/86 | 67/71 | 100 | 90 |
| 10/32 | 100/98 | 100/95 | 100/93 | 67/71 | 100 | 90 |

**On jay the two are level, and mWW is ahead on run-on** - 91 against 84 at 4/32, 93
against 86 at 6/32 - which is the surprise, because the mWW corpus contains **no run-on
positives at all**. Run 17 in `tuning.md` found the same effect in the other direction
(improving phrase-alone positives moved run-on detection by 20 points when it was
predicted not to move at all). Two independent observations now say run-on detection is
not primarily taught by run-on clips.

**On jen and ryan it is clearly worse**, and jen is the worst: 50% against 100% at
4/32, still only 90% at 8/32. jen is 10 clips, so each is 10 points - but the gap is
larger than that granularity and holds across every matched point.

### The score floor: microWakeWord has no usable range below 0.25

The single most important operational finding, and it is not visible in the ROC file.

| threshold | jay plain/run-on | adv FA | ordinary FA |
|---|---|---|---|
| 0.95 | 97 / 68  | 3/32  | 1/68  |
| 0.80 | 97 / 88  | 3/32  | 1/68  |
| 0.70 | 97 / 91  | 4/32  | 2/68  |
| 0.50 | 97 / 93  | 8/32  | 2/68  |
| 0.35 | 97 / 95  | 8/32  | 3/68  |
| 0.25 | 100 / 95 | 10/32 | 10/68 |
| 0.15 | 100 / 100 | **32/32** | **68/68** |

**Every negative in the corpus fires at 0.15 and below.** The resting score of this
model is 0.245 - `eval/eval_model.py` reports exactly that as the median for every negative
category - so a cutoff under it accepts everything. openWakeWord's operating point is
0.15 and `tuning.md` sweeps to 0.01; carrying either habit across would produce a
detector that never stops firing. **microWakeWord's usable range is 0.25 upwards.**
`eval/compare_models.py`'s sweep was extended to 0.95 for this reason.

Score resolution is 0.00078 (an int8 output, 256 levels, averaged over a window of 5),
so sweeping finer than that measures quantization rather than the model.

### Where mWW is actually better: high-precision operating points on jay

At 0.80 it reads jay **97/88 at 3/32 adversarial and 1/68 ordinary**. `d1bb9f4` needs
0.95 to reach 3/32, and there reads 97/75. On the adult speaker, at the precision you
would actually want on a device, the microWakeWord model is the better detector.

`extend` remains unsolved on this side too - 8/32 at 0.5, the same 6-8/32 band every
openWakeWord run since run 6 has sat in. It is not a frontend problem.

### Latency: 100 ms against 70 ms, both inside the gate

Measured on jay at matched precision (oww at 0.15, mWW at 0.50 - both ~7-8/32
adversarial and 2/68 ordinary):

| | median | p90 |
|---|---|---|
| `d1bb9f4` oww | 70 ms | 170 ms |
| mWW run 1 | 100 ms | 181 ms |

Both pass the 120 ms gate. mWW is 30 ms slower at the median, which is the price of a
1500 ms window scored every 30 ms against a 2000 ms window scored every 80 ms - and it
is small enough not to be a deployment argument either way.

### The shipped manifest could not fire, and nothing downstream would have caught it

`hey_seeree.json` shipped with `probability_cutoff: 1.0`, at which this model detects
nothing. The cause is in `train/mww/manifest.py`: `--max-faph` defaults to 0.0, and the only
ROC row with faph 0.000 is the **synthetic terminator** microWakeWord's
`generate_roc_curve` appends at (faph 0, frr 1) to close the curve when no measured
cutoff reaches the floor. It is not an operating point; it is a plotting artifact, and
it won the default every time.

Fixed by discarding rows with `frr == 1.0` before choosing. The default now refuses
loudly - "no cutoff achieves faph <= 0.0 while detecting anything" - instead of
emitting a dead manifest. **A cutoff of 1.0 is a legal value and the runtime loads it
without complaint**, so this would only ever have surfaced as a device that ignored
you.

### What to change next, in order

1. **Run-on positives in the corpus.** Absent entirely, and on the openWakeWord side
   they took held-out run-on detection from 5% to the 80s. Their cut point needs word
   timestamps Wyoming does not provide, which is the open problem.
2. **Kokoro in the mWW corpus.** Piper-only today. Run 17 measured two engines beating
   one by the largest margin since run 10.
3. **The child and jen gaps.** `add_child_range_copies` is the obvious lever and open
   question 1 in plan.md - whether it survives a 40-feature 10 ms frontend and int8
   quantization is now worth answering, because ryan and jen are where the loss is.
4. **`sampling_weight` / `penalty_weight` on the adversarial negatives.** ~1100 clips
   against pre-generated ambient sets, untuned, and `extend` is at 8/32.

### Not measured yet

- **Alignment.** `eval/check_model_alignment.py` is openWakeWord-only and its framing does
  not transfer (plan.md trap 4): it sweeps phrase placement in a 2000 ms window, and
  this is a streaming detector over 1500 ms with a sliding-window average. The latency
  above is the deployed quantity and is measured; where in its window the model wants
  the phrase is not.
- **`d1bb9f4` on the deployment runtime.** See the asymmetry noted at the top.
- **A long recording of the actual room.** The 68 ordinary negatives are a few minutes
  of TTS. A cutoff of 0.25 shows 10/68 there, which is already disqualifying, but the
  gap between 0.5 and 0.8 needs real background audio to choose between.
