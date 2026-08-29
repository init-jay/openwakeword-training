# Tuning guidance for wake-word training

Derived from measuring `hey_seeree.onnx` against 56 real recordings and a 100-clip
synthetic negative corpus. Every number below is measured, not estimated; the method
is at the end so it can be re-run after a retrain.

Priorities are ordered by measured impact per unit of effort, not by how interesting
they are.

---

## Held-out real recordings: the first honest numbers

Recorded after run 4 and run 5 had trained, so no model has seen them. 35 plain
clips and 57 run-on utterances ("hey seeree, what's the time" spoken as one breath),
hand-checked to remove fragments where the segmenter had split the phrase from its
command.

| | run 2 | run 4 | run 5 |
|---|---:|---:|---:|
| held-out plain (35) | 22/35 (63%) | **29/35 (83%)** | 28/35 (80%) |
| held-out run-on (57) | **3/57 (5%)** | **26/57 (46%)** | 16/57 (28%) |
| trained-set plain (56) | 51/56 (91%) | 53/56 (95%) | 54/56 (96%) |
| synthetic `cmd_run` | 30/36 (83%) | - | 36/36 (100%) |

**The run-on positives work, and the effect is far larger than the synthetic corpus
showed.** Run 2, trained before run-on positives existed, detects 3 of 57 real run-on
utterances - a near-total failure that the synthetic `cmd_run` sweep scored at 83%.
Adding run-on positives took it to 46%. The synthetic measure understated the problem
by an order of magnitude and overstates the fix: 100% synthetic against 46% real.

**The trailing margin matters most for real run-ons.** Run 5 scores 28% where run 4
scores 46%, on identical clips. That is the zero-margin regression showing up exactly
where the theory says it should - a model that fires before the word ends has nothing
left to distinguish "the phrase finished" from "the phrase continued into a command".

**Every previous positive number was measured on training data.** Held-out plain sits
at 83% where the trained set reads 95%. Treat the ~95% figures in the sections below
as training-set accuracy, not detection rate.

Gates worth adopting, all on held-out data:

- held-out plain detection: >= 90% (best so far 83%, run 4)
- held-out run-on detection: >= 80% (best so far 46%, run 4)
- and keep scoring `extend`/`hey_other` false accepts on the synthetic corpus, which
  is adversarial by design and still the right tool for that job

---

## Run 9 (ready): widen the positive speed range

The last measured failure with no fix applied. A synthetic sweep of run 4 detected
6/6 up to 1.25x, then 3/6 at 1.40x and 2/6 at 1.60x - training rendered nothing above
1.3x, so the model fails just outside the range it saw, while remaining fine below it
(0.55x gave 6/6). The asymmetry says widen the top only.

    PLAIN_SPEEDS  (0.7, 1.3) -> (0.7, 1.6)
    RUNON_SPEEDS  [0.8, 0.9, 1.0, 1.1, 1.2] -> [0.8, 1.0, 1.2, 1.4, 1.6]

Both move together because they are one variable. Covering fast phrase-alone
renderings while leaving run-ons at 1.2x would leave fast run-on speech - the
commonest real case - untrained.

Kokoro's 1.6x output was checked before committing to it: durations scale correctly
(1.4-1.6x), both words present in the timestamps, levels intact. "hey seeree" renders
in 390-590 ms there, which is the same range as the fast real clips run 4 missed.

**Not strictly single-variable.** The real corpus grew since run 8 - ryan 35 -> 42 and
a new speaker, jen, with 8 clips. That is +7.7% of the real set against an expected
large speed effect, so it should not obscure the result, but a *third speaker* is a
categorical change and anything it moves will be hard to attribute.

| | run 7 | run 9 succeeds if |
|---|---:|---|
| synthetic speed 1.40x / 1.60x | 3/6, 2/6 | **6/6, >= 4/6** |
| held-out plain | 89% | holds >= 85% |
| held-out run-on | 56% | holds >= 50% |
| `extend` + `hey_other` FA | 7/32 | does not worsen |

Watch false accepts particularly: faster positives are shorter, and a shorter phrase
is acoustically closer to the first syllable of "hey serious". This change could trade
against the near-miss problem.

---

## Run 8: `7aad08e` — the weight works, but only as a threshold in disguise

`max_negative_weight` 2000 -> 4000, single variable.

At threshold 0.5:

| | run 7 | run 8 |
|---|---:|---:|
| `extend` + `hey_other` FA | 7/32 | **3/32** |
| held-out plain | **89%** | 77% |
| held-out run-on | **56%** | 37% |

False accepts beat the < 4/32 target; both detection floors failed (>= 85% plain,
>= 50% run-on). By the pre-registered criteria, run 8 fails.

**But comparing at a fixed threshold conflates model quality with operating point.**
Detection at matched false-accept counts, threshold tuned per model:

| FA | run 7 plain/run-on | run 8 plain/run-on |
|---|---|---|
| 2/32 | **74% / 25%** | 60% / 16% |
| 3/32 | 77% / 26% | 77% / **37%** |
| 4/32 | **83%** / 37% | 77% / **42%** |
| 5/32 | **83%** / 39% | 77% / **44%** |

Neither dominates. Run 8 is better on run-ons at loose operating points, run 7 better
on plain throughout and clearly better at the tight 2/32 end the gate asks for. Most
of what the weight bought was movement along the same trade-off curve - which the
detection threshold provides for free, without retraining.

**Conclusion: revert to 2000.** Use the threshold to choose the operating point.

**A correction:** it was predicted above that the weight would not reach near-misses,
because openwakeword's auto-escalation is driven by a general-speech validation set.
It did reach them (7/32 -> 3/32). That reasoning confused the auto-escalation with the
base weight, which applies to every negative including the confusables.

**The < 2/32 gate looks unreachable at usable detection rates.** The best any model
manages there is 74% plain / 25% run-on. The gate was set from a single measurement of
the original model and against a corpus that is adversarial by construction - a fifth
of it phrase-extending. It should probably be restated as a rate the deployment can
actually tolerate, chosen alongside a threshold, rather than treated as a target to
train toward.

---

## Run 8 setup: `max_negative_weight` 2000 -> 4000

The first experiment aimed directly at `extend`, which has failed every run since run
2 and has only ever moved as a side effect of margin changes. `RUNON_TAIL_MS` stays at
(150, 300) and the real-sample corpus is unchanged, so this is single-variable again.

`max_negative_weight` is the end of a linear ramp - the negative-class loss weight
grows from 1 to it across training (`openwakeword/train.py:274`) - so raising it
penalises false positives harder. 4000 matches openwakeword's own escalation step.

**A caveat that may cap what this can achieve.** openwakeword auto-doubles the weight
between training sequences when `best_val_fp` exceeds `target_false_positives_per_hour`
(`train.py:291`), but `best_val_fp` is measured on the ACAV100M general-speech
validation set - where this model already scores 0/36. The training loop never sees a
phonetic near-miss, so its automatic tuning has nothing to push against. That is a
plausible reason `extend` has been immovable, and if so, a bigger weight will make the
model more conservative everywhere without specifically fixing near-misses.

| | run 7 | run 8 succeeds if |
|---|---:|---|
| `extend` + `hey_other` FA | 7/32 | **< 4/32** |
| held-out plain | 89% | **>= 85%** |
| held-out run-on | 56% | **>= 50%** |

The detection floors matter as much as the false-accept target: a heavier negative
weight buys precision with recall, and a model that reaches 2/32 by dropping run-on
detection to 30% is worse, not better.

If false accepts barely move while detection falls, the near-miss problem is not
reachable through loss weighting and the next lever is the wordlist - more neighbours
of the four persistent failures ("hey seriously", "hey series", "hey cereal", "hey
searing pain"), still disjoint from the eval corpus.

---

## Run 7: `23a1faa` — margin confirmed for run-ons, not for false accepts

| | run 4 | run 5 | run 6 | run 7 |
|---|---:|---:|---:|---:|
| effective margin | ~200 ms | ~50 ms | ~140 ms | ~225 ms |
| held-out plain (35) | 83% | 80% | **91%** | 89% |
| held-out run-on (57) | 46% | 28% | 40% | **56%** |
| `extend` + `hey_other` FA | **6/32** | 12/32 | 8/32 | 7/32 |
| median latency | 77 ms | -20 ms | **48 ms** | 83 ms |
| alignment band | 120-320 | 0-280 | 0-440 | 80-440 |

**Run-on criterion passed decisively.** Run 6 and run 7 share an identical real-sample
corpus and differ in one constant, so this is clean: **+85 ms of margin bought +16
points of real run-on detection** (40% -> 56%).

**False-accept criterion failed, and the trend it was testing looks confounded.**
Across margins 50 -> 140 -> 225 ms the count went 12 -> 8 -> 7: improving but
plateauing, not tracking. Run 4's 6/32 and run 7's 7/32 differ by one clip on a
32-clip set, and run 4 had half the real data. The earlier monotonic reading was
mostly the margin escaping the pathological zero case, not a real gradient.

**The band's lower edge lifted off zero** (0 -> 80 ms), which is what was wrongly
predicted for run 6 - the margin does control it, but run 6's 80 ms floor was not
enough to move it.

**The cost is latency**, 48 -> 83 ms, landing on the early-warning line. The alignment
peak is now 280 ms, which `check_model_alignment.py` flags as 50 ms beyond where
trimmed clips can reach. Further margin buys run-on detection against the 120 ms
latency gate, and there is not much room left.

**Run 7 is the best model so far** and the ship candidate: best run-on by 10 points,
plain within one clip of run 6's best, false accepts within one clip of run 4's best,
latency comfortably inside the gate.

Remaining: `extend` has failed every run since run 2's 4/32 and has only ever moved as
a side effect of margin changes. It needs an experiment of its own rather than more
margin tuning.

---

## Run 7 details: `23a1faa`

`RUNON_TAIL_MS (80, 200) -> (150, 300)`. Verified as the only functional line changed
against run 6, and the real-sample corpus is identical (160 jay + 35 ryan, holdout
untouched at 35 plain + 57 run-on) - so unlike runs 5 and 6 this is a genuine
single-variable experiment.

Effective margin lands around 225 ms, just past run 4's accidental ~200 ms, to test
whether the optimum is above it.

| | run 6 | confirms the trend if |
|---|---:|---|
| held-out run-on | 40% | **> 46%** (beats run 4) |
| `extend` + `hey_other` FA | 8/32 | **< 6/32** (beats run 4) |
| held-out plain | 91% | holds ~91% |

If run-on and false accepts both improve, the margin relationship is real and the
optimum is above 200 ms. If either is flat or worse, the three-run trend was
confounded by run 4's smaller real-sample set and this belongs back at (80, 200).

Watch held-out plain and latency too, which the trend table does not cover: a larger
margin puts run-on positives earlier in the window, and if that drags the alignment
late it would show up as plain detection falling and latency rising - the run 3
failure mode in milder form. Latency above ~80 ms is the early warning.

---

## Run 6: `3083d45` — best plain detection, and the margin prediction half failed

| | run 2 | run 4 | run 5 | run 6 |
|---|---:|---:|---:|---:|
| held-out plain (35) | 63% | 83% | 80% | **91%** |
| held-out run-on (57) | 5% | **46%** | 28% | 40% |
| `extend` + `hey_other` FA | 4/32 | 6/32 | 12/32 | 8/32 |
| median latency | 70 ms | 77 ms | -20 ms | **48 ms** |
| alignment band | 80-240 ms | 120-320 ms | 0-280 ms | 0-440 ms |

**Prediction failed on its own diagnostic.** The firing band was predicted to lift off
0 ms once `RUNON_TAIL_MS` was floored at 80 ms. It did not - the band still starts at
0 ms and got *wider*. The peak median rose from 0.960 to 0.989, so this is a more
confident model firing across a broader range, not one ignoring the margin. "Fires at
0 ms" conflates *learned to need no margin* with *confident enough to fire anyway*,
and is not the clean diagnostic it was treated as.

**Predicted correctly:** false accepts improved (12/32 -> 8/32), and run-on detection
recovered (28% -> 40%).

**What the runs actually show,** ordered by the effective trailing margin their run-on
positives carried:

| | effective margin | held-out run-on | extend FA |
|---|---:|---:|---:|
| run 5 | ~50 ms | 28% | 12/32 |
| run 6 | ~140 ms | 40% | 8/32 |
| run 4 | ~200 ms | 46% | 6/32 |

Monotonic on both. More trailing margin gives better real run-on detection *and* fewer
false accepts. Run 4's ~200 ms came from the +153 ms estimate bias - the bug that got
"fixed" was closer to optimal than either deliberate setting since.

Note run 6 also carries ~2x the real recordings, which is the likelier cause of the
jump in plain detection (83% -> 91%) and cannot be separated from the margin change.

**Next experiment:** `RUNON_TAIL_MS = (150, 300)`, holding everything else fixed. If
run-on detection exceeds 46% and false accepts fall below 6/32, the margin relationship
holds and the optimum is above 200 ms. If it does not, the trend was confounded by the
real-data differences and the margin should be left where it is.

---

## Run 6 details: `3083d45`

Two changes against run 5, one of them not visible in git:

* **`RUNON_TAIL_MS` (0, 100) -> (80, 200)** - the only corpus-affecting line in
  `train.py`. This is the experiment.
* **Real samples 91 -> 195** (jay 56 -> 160, ryan 35). `my_real_samples/` is
  gitignored, so a commit hash does not pin it. Real clips go from ~2% to ~4.4% of
  the positive set.

The second is a confound. If the false-accept count improves, it will not be provable
that the margin did it rather than the extra real data - though the *alignment band*
is diagnostic either way, since real samples do not move the run-on cut point.

Predictions, so the result can falsify rather than just be interpreted:

| | run 5 | run 6 expected |
|---|---|---|
| firing band lower edge | 0 ms | ~80 ms |
| `extend` + `hey_other` FA | 12/32 | toward 4/32 |
| median latency | -20 ms | ~70-80 ms (worse, and correct) |
| clean positives | 54/56 | possibly slightly lower |

If the band still reaches 0 ms, the margin theory is wrong and nothing else should be
changed until that is understood.

Latency getting *worse* is the intended outcome here. Run 5 bought its -20 ms by
firing before the word finished, which is what the false accepts were.

---

## Run 5: `705c23b` — the trailing margin went to zero

| | run 2 | run 4 | run 5 | gate |
|---|---:|---:|---:|---|
| alignment peak | 160 ms | 160 ms | 120 ms | ~180 ms |
| **fires from** | 80-240 ms | 120-320 ms | **0-280 ms** | |
| median latency | 70 ms | 77 ms | **-20 ms** | < 120 ms **PASS** |
| clean positives (trained set) | 51/56 | 53/56 | **54/56** | >= 55/56 |
| detection, command after | 50/56 | 53/56 | 53/56 | >= 90% **PASS** |
| **`extend` + `hey_other` false accepts** | 4/32 | 6/32 | **12/32** | < 2/32 |

Best positives and lowest latency of any run, and **three times the false accepts**.
The model scores 0.891 with the phrase flush against the window edge - it fires with
no trailing context at all.

**Cause: `RUNON_TAIL_MS` started at zero.** Once the timestamp cut made the boundary
exact, a tail of 0 ms produced a clip ending precisely where the wake word ends. Plain
positives always carry ~80 ms after the phrase (30 ms trim pad + ~50 ms residual), so
they never demonstrate a zero margin; run-on positives, at 40% of the set, now did.

Priority 3 called this in advance: *"an argument for a small trailing margin, not a
zero one - the margin is what lets the model hear that the word ended rather than
continued, which is exactly the discrimination Priority 1 is trying to teach."* The
margin is not padding around the phrase, it is evidence that the phrase finished.

**Fixed** by setting `RUNON_TAIL_MS = (80, 200)`, so run-on positives carry at least
the same ~80 ms floor that plain positives do.

**Also learned: the clean-positive gate has always been measured on training data.**
`train.py` trains on `my_real_samples/jay` and the gate reads the same directory. On
99 clips recorded after run 4 trained, that model detects **54/99 (55%)**, not the
95% the gate reported - and the gap is not explained by speed, since held-out clips
in the 600-800 ms band detect at 56% where trained clips of the same length hit 100%.
Future runs should be scored against a set recorded after training.

---

## Run 5 details: `705c23b`

Two changes against run 4:

* **Timestamp-exact run-on cut.** The boundary now comes from Kokoro's
  `/dev/captioned_speech` word timestamps instead of a phrase-alone estimate,
  removing a median +153 ms bias that was also voice-dependent (`af_bella` ~0 ms,
  `bf_lily` +348..+459 ms) and occasionally cut inside the wake word.
* **Parallel TTS generation.** Throughput only - but note it changes the *corpus*,
  not just the speed: speeds are drawn up front in the main thread, so the sequence
  of random draws differs from the sequential version even at the same seed. Not a
  bias, just different clips.

Speed range deliberately unchanged at U(0.7, 1.3), so speed 1.40/1.60 should still
fail. That keeps it available as the next single-variable experiment.

What to expect:

* firing band tightening from 120-320 ms back toward 80-240 ms, as the +153 ms bias
  on 40% of the positives goes away
* `cmd_run` holding at 36/36 - only where the run-on clips are cut changed, not why
  they exist
* `extend` false accepts *possibly* recovering some of the 4/32 -> 6/32 regression,
  because the trailing region no longer carries an extra ~150 ms of command speech.
  A mechanism, not a prediction; 6/32 holding would be an equally sensible result.

---

## Run 4: `9151908` — alignment recovered, Priority 3 solved

| | run 2 | run 3 | run 4 | gate |
|---|---:|---:|---:|---|
| alignment peak | 160 ms | 480 ms | **160 ms** | ~180 ms |
| fires from | 80-240 ms | 280-560 ms | 120-320 ms | |
| median latency | 70 ms | 130 ms | **77 ms** | < 120 ms **PASS** |
| detection, command after (real) | 50/56 | 54/56 | 53/56 (95%) | >= 90% **PASS** |
| synthetic `cmd_run` | 30/36 (83%) | - | **36/36 (100%)** | |
| clean positive detection | 51/56 | 53/56 | 53/56 (95%) | >= 55/56 |
| `extend` + `hey_other` false accepts | 4/32 | 7/32 | 6/32 (19%) | < 2/32 |

**Two gates now pass.** The alignment recovered completely despite this run still
carrying the v2 cut's +153 ms bias - the 60% of positives that are phrase-alone
dominate the mode. The bias shows up instead as a shifted band: firing moved from
80-240 ms to 120-320 ms, about 40 ms later at both edges, which is the residual to
expect the timestamp-exact cut to remove.

**Priority 3 is solved.** Run-on utterances rendered as one breath went 83% -> 100%.
That was the point of the run-on positives, and it worked.

**The cost is the `extend` count**, 4/32 -> 6/32 against run 2. This is the trade
flagged when the change went in: positives that end in command speech make the
trailing region less discriminating, and trailing context is what separates the
phrase from its extensions. Two clips, for +17 points on run-on detection.

**Untouched, as expected:** speed 1.40 (3/6) and 1.60 (2/6). The speed range is still
U(0.7, 1.3).

**New finding - one voice carries most of the synthetic misses.** `af_nova` fails 13
of its 31 clips; every other voice fails 2-4. It passes at -06 and -12 dBFS and fails
from -18 down, so it is marginal rather than broken, and it is in the training set.
It sits in `VOICES[:6]`, which is why the level, noise and speed sweeps all read 5/6.
Worth listening to before reading much into any single sweep row.

### What that commit contained

* Run-on positives cut by the **v2 estimate** — `cut = phrase_len - pad + tail`,
  `RUNON_TAIL_MS = (0, 100)`. Measured afterwards at a median **+153 ms late** and
  voice-dependent (`af_bella` ~0 ms, `bf_lily` +348..+459 ms), with 2 of 18 sampled
  clips cutting slightly *inside* the wake word. Better than run 3's 270-470 ms, but
  not the fix.
* All three openWakeWord patches, and `onnxruntime-gpu` in the Dockerfile.

Not in it, so **not** in this run: the timestamp-exact run-on cut
(`/dev/captioned_speech`), parallel TTS generation, and multi-server TTS.

So run 4 tests one thing: whether reducing the run-on overshoot from ~370 ms to
~150 ms brings the alignment peak back from 480 ms. Expect an improvement but not a
full recovery to run 2's 160 ms, and read a residual as the remaining +153 ms rather
than as the approach failing.

**Whether it got GPU feature computation depends on when the image was built, not on
the commit.** The tell is in its own log: `Computing features` at ~2.46 it/s is the
single-threaded CPU path; substantially faster means the GPU is in use.

---

## Run 3: run-on positives, more data — and an alignment regression

`hey_seeree_ee215d8e...onnx`, the first run with run-on positives, `samples_per_voice`
300 and `augmentation_rounds` 3:

| | run 2 | run 3 | gate |
|---|---:|---:|---|
| `extend` + `hey_other` false accepts | 4/32 | 7/32 | < 2/32 |
| clean positive detection | 51/56 | 53/56 | >= 55/56 |
| detection, command immediately after | 50/56 | **54/56 (96%)** | >= 90% **PASS** |
| median latency | 70 ms | 130 ms | < 120 ms |
| alignment peak | 160 ms | **480 ms** | ~180 ms |

**The run-on positives worked.** The command gate passes for the first time, 89% -> 96%.

**But the cut point was wrong and it undid Priority 2.** The alignment peak went back to
480 ms, exactly where the original pre-trimming model sat. Two errors compounded, both
in `generate_runon_samples`:

* `phrase_len` came from `trim_silence`, which carries a 30 ms trailing pad that was
  treated as the phrase's end.
* The phrase is shorter inside the run-on than alone, because its final syllable is
  coarticulated into the next word. Measured across voices and speeds: **median 190 ms**
  (range 0-350 ms). The first version noted this and called it "the intent", then added
  U(50, 250) ms on top of it.

Together the cut kept **270-470 ms** of command audio where plain positives keep ~80 ms.
With `end_jitter` U(0, 200) on top, the phrase landed a median ~470 ms from the window
end. The positive set was bimodal and the model learned the later mode.

That also explains the false-accept regression. When the trailing region holds command
speech in the positives and the continuation of "hey serious" in the negatives, it stops
discriminating anything, so the model learns to ignore it - and trailing context is
exactly what separates the phrase from its extensions.

**Fixed** by subtracting the pad and cutting the jitter to U(0, 100) ms. The remaining
overshoot is the coarticulation difference, which cannot be measured per clip and is
left as the tail's natural variation. Deliberately not corrected further: a too-large
correction would cut into the wake word and corrupt the positive, which is far worse
than being slightly late. Verified on generated clips - median 839 ms, down from 965 ms.

Expect the next run to put the peak near 200 ms rather than 160, since run-on positives
still sit a little later than plain ones. That is the price of the command gate.

---

## Results after Priorities 1 and 2

`hey_seeree_aligment_fix.onnx`, measured against the old model on the same harness and
the same 56 positives (`eval_model.py`, threshold 0.5):

| | old | new | gate |
|---|---:|---:|---|
| `extend` + `hey_other` false accepts | 16/32 | **4/32** | < 2/32 |
| clean positive detection | 53/56 | 51/56 | >= 55/56 |
| detection, command immediately after | 46/56 | **50/56** | >= 27/30 (90%) |
| detection, 300 ms pause then command | 52/56 | 50/56 | — |
| median latency from end of speech | 220 ms | **70 ms** | < 120 ms |

The tflite conversion reproduces these numbers exactly — identical per-clip scores, not
merely identical totals — so the gates hold for the shipping artifact, not just the ONNX.

Three of the four moved substantially and latency now passes. Two things are worth
reading carefully:

**The negatives generalised.** The four surviving false accepts are "hey seriously",
"hey series", "hey cereal" and "hey searing pain" — the nearest phonetic neighbours of
the phrase, and none of them were in the training wordlist, which is disjoint from this
corpus by construction. 13/20 -> 4/20 on `extend` is a real generalisation gain, not
memorisation. `hey_other` is now 0/12.

**The "trained on quiet" signature survives — the spliced test just cannot see it.**
By that test the 300 ms pause, which used to recover six detections (46 -> 52), now
recovers none (50 -> 50), and this section previously concluded the problem was solved.
It is not. Rendering the phrase and the command as *one* TTS utterance
(`generate_positives.py --sweeps command`) puts the gap straight back:

| | detected |
|---|---:|
| `cmd_pause` — "hey seeree**,** what's the time?" | 35/36 (97%) |
| `cmd_run` — "hey seeree what's the time?" | **30/36 (83%)** |

The splice is the insensitive instrument. It keeps the phrase's *isolated* ending —
the final syllable released and decaying as it would in isolation — and abuts unrelated
audio after it. In real speech the "-ee" of "seeree" is coarticulated into "what's" and
shortened, so the phrase itself is acoustically different, not just its surroundings.
Priority 3 remains open, and should be measured with the single-utterance corpus.

**The cost:** clean positives went 53 -> 51, and the cause is not yet known.
`max_negative_weight` (`train.py`, 2000) is the lever if the model needs loosening.

Three explanations for the misses were checked and ruled out:

* *Misalignment in the recordings.* Measured against a threshold crackle cannot reach
  (-15 dB of peak RMS), trailing non-speech is at most 170 ms across all 91 clips,
  median 70 ms — inside the normal pad + jitter budget. No clip is misaligned.
* *Merged utterances.* The long clips (`jay/hey_seeree_0001.wav`, 1650 ms) are
  continuously energetic end to end — one slow utterance, not two.
* *Recording level or noise.* The five missed clips have a median peak amplitude of
  2842 against 2574 for the detected ones, and the same -25 dB noise floor. The five
  quietest clips in the corpus are all detected.

Worth knowing but not the cause: the corpus is quiet overall — median peak 2472 of
32767 (~ -22 dBFS) with a -26 dB noise floor, which is what makes it sound crackly.
That applies to every clip equally, detected or not.

### What actually degrades detection

Measured with `generate_positives.py` (126 synthetic clips) scored with
`eval_model.py --by-group`. These are training-distribution clips, so the sweeps that
push outside what training saw are the informative part; the flat parts are a floor.

| speed | 0.55 | 0.65 | 0.75 | 1.00 | 1.25 | 1.40 | 1.60 |
|---|---|---|---|---|---|---|---|
| detected | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | **3/6** | **2/6** |
| median score | 0.969 | 0.987 | 0.988 | 0.984 | 0.982 | **0.481** | **0.023** |

`train.py` renders positives at U(0.7, 1.3). The model holds up *below* that range —
0.55 still detects 6/6 — but falls off a cliff just above it. Fast delivery is the
untrained failure mode, and the asymmetry says the fix is widening the top of the range,
not the bottom.

**Recording level does not matter.** 6/6 detected at every level from -6 to -34 dBFS,
median score flat at 0.983-0.985. **SNR does**: 6/6 down to 15 dB, 5/6 at 10 dB, 4/6 at
5 dB. The real recordings sit at ~22-26 dB SNR, so they are clear of the degradation
point, and their -22 dBFS level costs nothing. Recording hotter is about margin, not
about a level the model needs. (Note the latency column is meaningless for the noise
sweep — added noise moves the 2%-of-peak speech-end marker.)

**This partly explains the five missed real positives.** Four of the five are fast:
`hey_seeree_0011.wav` at 300 ms is shorter than all 51 detected clips (shortest
detected: 430 ms), and three more sit at the 84th-94th percentile for speed. The fifth,
`hey_seeree_0016.wav` at 1440 ms, is slow, so speed is a strong risk factor rather than
a complete account — and n=5 is not much to conclude from.

Remaining work, in order:

1. Widen the positive speed range in `train.py` (`generate_kokoro_sample`, currently
   `np.random.uniform(0.7, 1.3)`) to about `(0.7, 1.6)`. Cheapest remaining change,
   and it targets a measured failure.
2. Priority 3 proper: positives where the phrase runs into a command as one utterance
   (83% vs 97% with a pause). Generating those as training data is the same TTS call
   `generate_positives.py --sweeps command` already makes.
3. The four `extend` false accepts, which is what Priority 3's trailing context is
   supposed to discriminate.

Both 1 and 2 push the positives to sound *more* like their neighbours, so watch the
`extend` count when either lands - it can trade against them.

---

## What the original model did

| | |
|---|---|
| detection rate, clean positives | 56/56 at threshold 0.5 |
| detection rate, phrase followed immediately by a command | **20/30 (67%)** |
| median latency from end of speech | 191 ms (at a 40 ms prediction step) |
| false accepts on "hey" + similar-sounding word | **18/100** |

Three problems, in priority order: it fires on things it should not, it misses a third
of naturally-spoken commands, and it fires ~190 ms late. All three are training-data
problems. None of them are inference problems — that side has been measured out.

---

## Priority 1: the negative phrase list is too small

`train.py:356` currently trains against nine negatives:

```python
negative_phrases = [
    "hello", "hi there", "good morning", "excuse me", "okay",
    "hey google", "alexa", "hey jarvis", "computer",
]
```

Scoring the model by category makes the consequence unusually clear:

| category | n | fired >=0.5 | in the training negatives? |
|---|---:|---:|---|
| other assistants ("hey google", "alexa") | 8 | **0** | yes |
| general conversation | 36 | **0** | effectively |
| bare commands, no wake word | 12 | **0** | effectively |
| siri-sounds in speech, no "hey" | 12 | **0** | no, but no "hey" |
| **"hey" + other name** ("hey Sarah", "hey Cindy") | 12 | **5** | **no** |
| **phrase-extending** ("hey serious", "hey series") | 20 | **13** | **no** |

The model rejects perfectly what it was trained to reject and fails on everything
adjacent that it never saw. "hey serious" scores **0.995**. This is not a subtle
generalisation failure; it is a gap in the wordlist.

**Fix (done):** the wordlist is now split in `train.py` into `BASE_NEGATIVES` (the
original nine — wake-word independent) and `CONFUSABLE_NEGATIVES[safe_name]`, 38
phrases for `hey_seeree` covering the three shapes that failed: the phrase continuing
into another word ("hey Serena", "hey season"), "hey" plus another name ("hey Sienna",
"hey Cynthia"), and the same sounds in running speech with no "hey". Bare "hey" is in
there too — it is what teaches that the second syllable is required. 47 phrases total.
`--negatives-file` supplies the confusable half for a wake word with no built-in entry,
and training without any confusables now warns.

**The training phrases are deliberately disjoint from the eval corpus below.** An
earlier draft of this section proposed "hey serious", "hey Sarah" and so on, which are
verbatim the `extend` and `hey_other` entries in `generate_negatives.py` — training on
them would have made the gates at the bottom of this file measure memorisation instead
of generalisation. Anything added to either list should be checked against the other.

One supporting fix: `generate_kokoro_samples` cycled the wordlist as `texts[i % len]`
from the same starting point for every voice, so with 47 phrases the negative *test*
set (20 clips per voice) only ever saw the first 20. It now offsets each voice's start,
giving all 47 phrases 285-286 train and 28-29 test renders. Total negative clip count is
unchanged, so the class balance is the same.

Cheapest change here by a wide margin, and it is the one that fixes the worst
behaviour. `max_negative_weight` (`train.py`, currently 2000) is the lever if the
model then becomes too conservative.

---

## Priority 2: the phrase sits too early in the window

The model was measured against how much audio has arrived since the phrase ended:

```
   t after phrase end   trailing context   median score
        0 ms                 0.00 s        0.001
      160 ms                 0.16 s        0.418
      240 ms                 0.24 s        0.975
      440 ms                 0.44 s        0.992   <- peak
      720 ms                 0.72 s        0.617
      800 ms                 0.80 s        0.053
```

It scores **0.001 with the phrase flush against the window edge** and needs ~200 ms of
trailing audio before it crosses 0.5. That wait *is* the latency — the model is not
computing during it, it is waiting for audio that does not exist yet.

The cause is the alignment `check_alignment.py` already documents: `create_fixed_size_clip`
puts the end of the *array* at the end of the window, so whatever trailing room tone a
clip carries decides where the phrase actually lands.

**But the numbers do not add up, and that is the first thing to check.** With trimming
on, `trim_silence` leaves `pad_ms=30` and `create_fixed_size_clip` adds
`end_jitter` from U(0, 200) ms — so a correctly trimmed clip should land the phrase
30-230 ms from the window end, median ~130 ms. The model behaves as if trained at
~440 ms. Something is inconsistent.

### Diagnosed: the measured model predates trimming

Four measurements, and they close the gap completely.

**1. The 440 ms is real, not a measurement artifact.** Re-measured in the *training*
framing rather than by streaming — one fixed 32000-sample window, features via
`AudioFeatures.embed_clips`, which is the exact call `compute_features_from_generator`
makes — sweeping where the phrase is placed, over the 91 real recordings:

```
   gap (phrase end -> window end)    0    120   160   200   400   640   720
   median score                   0.001 0.044 0.860 0.978 0.991 0.782 0.018
```

Peak at 400 ms, usable band roughly 160-640 ms, collapse below ~120 ms. The streaming
measurement said 440 ms; the offline one says 400. So the streaming feature path adds
no meaningful lag — this is a property of the model, learned from its training data.

**2. `trim_silence` works.** On Kokoro output at the same server and speed range, the
audio left after the phrase ends (measured at the -34 dB marker the latency method
uses, against the -40 dB threshold the trim uses) is a median of **50 ms**, p90 79 ms.
No hidden sub-threshold tail. The real recordings leave 0 ms.

**3. Untrimmed Kokoro explains 400 ms exactly.** Short Kokoro phrases as the server
emits them carry a mean of **304 ms** of trailing silence, median gap 317 ms. Add the
U(0, 200) ms jitter and the phrase lands a median of **~417 ms** from the window end.
That is the observed peak. Trimmed, the same clips predict 30 ms pad + 50 ms residual
+ jitter ≈ **180 ms**.

**4. The timeline agrees.** `trim_silence` landed in `c1f2e56` at 14:50 on 2026-08-27;
`hey_seeree.onnx` was written at 15:56 the same day — 66 minutes later, far less than a
run that makes 13,400 TTS calls and then trains for 50,000 steps. The run that produced
the measured model started before trimming existed.

**Conclusion: no code change needed.** The fix is already in `train.py` and takes effect
on the next retrain. Expect the peak to move from ~400 ms to ~180 ms and the latency to
drop by roughly the same amount, which clears the 120 ms gate on its own.

**Verify after the retrain:**

```bash
python check_model_alignment.py --model my_custom_model/hey_seeree.onnx
```

The peak should have moved from ~480 ms to ~180 ms, and the score at 120 ms should no
longer be near zero. `check_alignment.py my_custom_model/hey_seeree/positive_train` is
the cheaper check on the input side — run it after the "Trimming silence" step prints,
and it should report a mean gap under the 200 ms jitter.

**Target:** phrase ending 100-150 ms before the window end. Do not drive it to zero —
see the note under Priority 3.

**Watch for:** clips longer than the 2 s window are truncated to their *first* 2 s
(`data.py:676-677`) and the tail is discarded, so a long negative sentence only
contributes its opening. 4 of 32 sampled Kokoro sentences hit this. It is harmless when
the confusable sound is early in the sentence, which is why the running-speech negatives
added in Priority 1 put theirs in the first few words.

---

## Priority 3: it is trained on "wake word, then quiet"

| trailing context after the phrase | detected |
|---|---:|
| silence | 28/30 (93%) |
| **command speech immediately after** | **20/30 (67%)** |
| 300 ms pause, then command | 28/30 (93%) |

Speaking naturally — "hey siri what's the time?" — costs a quarter of detections. The
300 ms gap restoring it completely is the signature of a model that learned the phrase
is followed by quiet.

**Fix:** vary what occupies the trailing region in the positive examples. Roughly half
silence or room tone, half the onset of a command ("hey siri, what's the time",
"hey siri, turn on the lights"), so both interaction styles are represented. Put the
same command speech in the negatives without the wake word, or the model may learn
"speech after ≈ wake word".

This interacts with Priority 2 in a useful way: if the model fires with the phrase at
the window edge, the command has not arrived yet and cannot interfere at all. That is
an argument for a *small* trailing margin, not a zero one — the margin is what lets the
model hear that the word ended rather than continued, which is exactly the
discrimination Priority 1 is trying to teach.

Note that trailing context is currently doing **no** discriminative work: "hey serious"
peaks at 0.994 with the whole word inside the window. Today the 440 ms buys latency and
nothing else. After Priority 1 it becomes load-bearing, which is why the margin should
shrink rather than vanish.

---

## Suggested order

1. ~~Extend `negative_phrases` (Priority 1).~~ Done — retrain and re-measure false accepts.
2. ~~Run `check_alignment.py` on the positives and reconcile the 440 ms (Priority 2).~~
   Diagnosed — the measured model predates trimming, and the retrain fixes it with no
   code change. Confirm the peak moved once the new model exists.
3. Add command-following positives and matching negatives (Priority 3).
4. Only then consider shrinking the model's receptive field (fewer than 16 embedding
   frames), which is the secondary latency lever and changes `input_shape`.

Steps 1-3 are all wordlist and alignment changes to an existing pipeline. None require
touching openWakeWord itself.

---

## Verifying a retrain

Generate the negative corpus (100 clips across six categories, ~4 minutes against a
local Kokoro server):

```bash
python generate_negatives.py \
    --url http://192.168.2.14:8880/v1/audio/speech \
    --out negatives_tts
```

Then measure both sides:

```bash
python eval_model.py \
    --model my_custom_model/hey_seeree.onnx \
    --positives my_real_samples/jay \
    --negatives negatives_tts
```

`eval_model.py` streams the model over each clip the way live detection does and
scores all four gates below. It reproduces the original measurements on the old model
to within a few points — 93% detection with a 300 ms pause against 93% measured, 220 ms
median latency against 191 ms, 16/32 adversarial false accepts against 18/32 — except
clean positives, where it finds 53/56 for a model originally measured at 56/56. Read
clean-positive counts as a comparison between models under this harness, not against
that 56/56.

Read the negatives **per category**, not pooled — the corpus is adversarial by
construction (a fifth of it is phrase-extending), so a pooled false-accept rate is
meaningless. `general` is the realistic background rate; `extend` and `hey_other` are
the ones that should improve.

Gates worth holding a retrain to:

- `extend` and `hey_other` false accepts at 0.5: **< 2/32** (currently 18/32)
- clean positive detection at 0.5: **>= 55/56** (currently 56/56 — do not regress it)
- detection with a command immediately following: **>= 27/30** (currently 20/30)
- median latency from end of speech: **< 120 ms** (currently 191 ms)

### Method behind these numbers

Latency is measured by streaming each clip with a noise floor rather than digital
silence — pure zeros are a pathological input for the melspectrogram and shift scores.
"End of speech" is the last sample above 2% of peak amplitude. Latency is the audio
offset at which the score first crosses 0.5, minus that marker. Detection with a
following command is simulated by concatenating unrelated speech directly onto the
trimmed phrase.
