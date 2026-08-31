# Tuning notebook for wake-word training

**This is a lab notebook, not reference documentation.** One section per training
run, newest first, each recording the hypothesis, the commit it ran on, and what the
measurement said - including the predictions that turned out wrong. It is kept in
that form deliberately: several conclusions here were reversed by later runs, and the
reasoning is worth more than the conclusions when picking the next lever.

Twelve runs against "hey seeree", every number measured rather than estimated.

## Current settings and results

    --training-steps 50000        50k beat 100k on run-on, replicated
    --real-copies 10              the single biggest lever found
    --runon-fraction 0.4          positives that run into a command
    --max-negative-weight 2000    4000 traded detection for precision, no net gain
    --samples-per-voice 300
    --augmentation-rounds 3
    RUNON_TAIL_MS = (150, 300)    trailing margin; must not reach zero
    PLAIN_SPEEDS  = (0.7, 1.6)

**Ship candidate: `9a938fb`** — see the batched-TTS section for why it is preferred
over `eea1c56` and `41c5cbc`, which it does not measurably beat. Tune the threshold
before deploying; 0.5 is the wrong operating point.

Best models, on recordings made after they trained, at 8/32 adversarial false
accepts. **These are jay-only numbers, and jay is an adult.** The second speaker
(ryan, aged 4) reads 24% plain on the same models - see run 12:

| | |
|---|---|
| held-out plain (35 clips) | **99%** |
| held-out run-on (57 clips) | **95%** |
| median latency from end of speech | ~50 ms |

Where it started: 63% plain, **5%** run-on, 13/20 false accepts on "hey serious",
220 ms latency, 83 minutes per run (now ~16).

## Four rules, learned the expensive way

**1. Score on recordings made AFTER the model trained.** `train.py` trains on
everything under `my_real_samples/`, so pointing an evaluation there reports training
accuracy. It overstated detection by ~10 points and hid a much larger gap on run-on
speech. Eight runs were judged on contaminated numbers before this was noticed.

**2. Compare models at matched false-accept rates, never at a fixed threshold.** Two
runs of an identical configuration read 77% and 67% run-on at threshold 0.5 and both
reach 95% at 8/32 false accepts. What varies between runs is where the score
distribution sits, not how well the model separates classes. Several conclusions in
the sections below were drawn at a fixed 0.5 and are unreliable for that reason; the
large effects survive re-checking, the few-point ones do not.

**3. Synthetic evaluation is a lower bound on difficulty, not a gate.** A model
scoring 100% on synthetic "wake word + command" clips detected 46% of real ones. An
earlier splice-based test showed no problem at all. The synthetic speed sweep, at six
voices per point, measures which voices are hard rather than which speeds.

**4. Score every speaker separately, and never pool them.** Added after run 12, which
found the second speaker - a 4-year-old - at 24% detection while the pooled-by-absence
headline said 99%. The holdout had no child clips in it for eleven runs, so the gap
could not appear. A speaker missing from the holdout does not make the number worse,
it makes the number meaningless.

The one thing never solved: **false accepts on close phonetic neighbours** ("hey
serious", "hey series"). 13/20 at the start, 6-8/32 since, and no lever tried has
moved it much. Real recordings of near-misses, spoken by the actual users in the
actual room, are the untried idea most likely to help - by symmetry with real
positives, which turned out to matter far more than their 4% share of the corpus.

---

## Corpus bug, runs 1-12: six voices say the wrong word

Six of the 42 English voices do not say "hey seeree": **af_alloy, am_echo, bf_alice,
bf_lily, bm_daniel, bm_fable**. Judged by ear over every voice rendering the phrase
once (`vtlp_demo/voices/`, one file per voice). Now in `MISPRONOUNCING_VOICES`, keyed
per wake word, and excluded from positives and negatives alike.

A wake word worth having is not a dictionary word, so Kokoro's g2p has to guess at it,
and voices guess differently. Every clip from a bad voice is a mislabelled positive.
At 1/42 of the voice list each that is ~2.4% of the Kokoro corpus per voice, **~14%
for the six**, across plain and run-on alike since both draw from the same list.

Together with the uppercase bug below, roughly **a fifth of the synthetic positives
in every run to date were not the wake word.**

### Duration is not a proxy for pronunciation - it failed twice

Worth recording because it was tried twice and confidently gave the wrong answer both
times:

| | duration | vs median | verdict |
|---|---:|---:|---|
| `bm_fable` | 1121 ms | **1.00x** | **wrong** - the single most median-length voice in the set |
| `am_echo` | 1273 ms | 1.14x | **wrong**, inside the normal spread |
| `af_alloy` | 1602 ms | 1.43x | wrong, and flagged |
| `bf_lily` | 1520 ms | 1.36x | wrong, and flagged |
| `af_v0sky` | 938 ms | 0.84x | fine |
| `am_onyx` | 872 ms | 0.78x | fine |

The same proxy also cleared "HEY SEEREE" as emphatic delivery when it was being
spelled out (+12% duration). Two failures, opposite directions: it flagged correct
voices and cleared incorrect ones. **Listening to all 42 takes a couple of minutes and
is the only method that works.** Do it for every new wake word.

None of the eleven `v0` variants were rejected, and no rejected voice has a `v0` twin
that was also rejected, so the fault is per-voice rather than per-speaker-family.

`--exclude-voices` adds to the list at the command line for a wake word with no
built-in entry.

---

## Corpus bug, runs 1-12: a sixth of the plain positives were spelled out

`positive_texts` contained `wake_word.upper()`. Kokoro renders "HEY SEEREE" as
**spelled-out letters** - "hey S-E-E-R-E-E" - and every run so far labelled that as
the wake word. With 6 text slots cycled evenly, that is **~1/6 of plain positives**,
or ~8% of all Kokoro positives once run-ons are counted. Removed.

It is uppercase on the *invented* word that does it. "HEY seeree" measures 0.031 from
plain in embedding space (nothing happens); "hey SEEREE" measures 0.030 from
"HEY SEEREE" (both spell it out). A real word in caps is fine - the wake word is not a
real word, which is exactly why it works as a wake word.

**Caught by ear. Two measurements had already looked at it and both said the wrong
thing**, which is the part worth keeping:

* **Duration**: 1083 ms against 965 ms, +12%. Spelling six letters should have
  doubled it. The change was read as emphatic delivery. A 12% difference was never
  strong enough to conclude anything, and it was used to *rule out* spelling.
* **Embedding distance**: 0.535 from plain - about the same as a completely different
  voice (0.70) and far above one speed-grid step (0.29). Read as "excellent
  diversity". It actually meant "this is not the same phrase."

**A large embedding distance cannot tell useful variety from a different utterance.**
Both look identical to that metric. It is a *screening* tool at best, and anything
added to `positive_texts` has to be listened to before it goes in.

The eval corpus is unaffected - `generate_positives.py` and `generate_negatives.py`
never used `.upper()` - so every gate and false-accept number in this file was scored
on correctly-pronounced audio. Only training positives were contaminated.

Unknown how much this cost. It is a mislabelled positive, so the model was taught
that a spelled-out variant is the wake word, which should hurt precision rather than
recall. It does not obviously explain the child gap, which has a sufficient cause
already.

### Replacement list

`.upper()` gone. `.lower()` gone too - it is the same string as `wake_word` for a
lowercase wake word, so 6 slots only ever held 5 strings. `.title()` gone as useless:
0.027 / 0.053 from plain, indistinguishable. What remains is punctuation, which moves
prosody without touching pronunciation - measured af_bella / am_adam against plain:

| variant | distance |
|---|---:|
| `hey seeree,` | 0.437 / 0.344 |
| `hey seeree!` | 0.291 / 0.201 |
| `hey seeree...` | 0.264 / 0.523 |
| `hey seeree!!` | 0.158 / 0.232 |
| `hey seeree?` | 0.105 / 0.304 |
| `hey seeree.` | 0.086 / 0.294 |

Seven slots, seven distinct strings, all pronounced correctly. Scale for reading those
numbers: a different voice is ~0.70, one step of `PLAIN_SPEED_GRID` is ~0.29, and a
plain re-request of the identical prompt is ~0.05 (Kokoro is not deterministic, but
repeats carry ~1/6 the novelty of one speed step - which is why raising
`--samples-per-voice` is the weakest diversity lever available).

---

## Run 13 (staged, not yet trained): child-range positives

Implements lever 1 from run 12. **Nothing below is measured** - it is what was built
and what it should do, written down before the run so the prediction can be wrong on
the record.

`add_child_range_copies()` in `train.py` adds pitch/formant-shifted copies of the
Kokoro positives after generation and before trimming. `--child-fraction` (default
0.5) sets how many clips get one; `--child-fraction 0` restores the old corpus.

* **Additive, not substitutive.** The adult clips all survive, so jay's density is
  untouched. Substituting would have bought ryan by spending jay, which run 10's
  result argues against.
* **Ratio per voice sex**, straight from the listening test: female 1.20-1.35x
  (-> 272-306 Hz, straddling ryan's 291), male 1.15-1.30x (-> 152-172 Hz, the gap
  between jay and ryan). Male voices are not asked to reach a child, because above
  1.30x they are audibly chipmunk and that is a cue the model would learn instead of
  the phrase.
* **Kokoro clips only.** Ryan needs no shifting; jay is male, so shifting him reaches
  a teen range that ~15 Kokoro male voices already cover far more cheaply.
* **Duration is preserved** - resample by R, then WSOLA back to the original length,
  so the shift changes the speaker and not the delivery speed. Speed is already an
  independent axis (`PLAIN_SPEED_GRID`) and conflating the two would confound them.
* The voice is now in the Kokoro filename (`kokoro_af_bella_<uuid>.wav`), which is
  how the pass knows a clip's sex.

Verified before committing: F0 lands where intended (`af_bella` 1.31x -> 296 Hz,
`am_adam` 1.25x -> 168 Hz), duration is preserved to the sample, real clips are
skipped, and the WSOLA output matches `ffmpeg -af asetrate,aresample,atempo` on the
same clips to within the F0 estimator's resolution. scipy only - the trainer image has
no ffmpeg.

**Prediction.** Ryan's held-out plain detection moves from 24%; jay's 97% does not
regress. `extend` false accepts are the thing to watch: the shifted clips are new
positive material near the phrase, so they could plausibly move that number either
way. If ryan improves and jay drops, the additive design failed and the fraction is
the lever.

**Score it per speaker.** A pooled number would hide exactly the failure this run
exists to fix.

---

## Run 12: `2213187` — the model is adult-only, and eleven runs of notes never said so

The run itself is minor. What it exposed is not: **the second speaker is a 4-year-old
child and the model detects him 24% of the time.** Every headline number in this file
is a measurement of one adult male. Skip to "the ryan holdout arrived" for that; the
first two sections are how the blind spot stayed hidden.

The code diff is nothing: `git diff 9a938fb 2213187` touches only `run-training.sh`
and the post-training reporting in `train.py` (the freshness check that replaced the
exit-code check, since openwakeword exits 1 on the tflite conversion after the .onnx
is already written). The corpus is what moved. `my_real_samples/` was re-scp'd to the
training VM with 17 new clips of **ryan**, recorded 20:18-20:23 on 30 Aug, half an
hour before this model was written:

| | `9a938fb` | `2213187` |
|---|---:|---:|
| jay | 160 | 160 |
| ryan | 42 | **59** |
| jen | 8 | 8 |
| total real clips | 210 | **227** (+8%) |

At `--real-copies 10` that is +170 weighted clips, and ryan's share of the real corpus
goes 20% -> 26%.

### The first pass at evaluating it was jay-only, so it measured none of that

At the time this run was first scored, `my_real_samples_holdout/` held 35 plain and 57
run-on clips, **all of jay**. `my_real_samples_holdout/ryan/` and `ryan_runon/` existed
and were empty - created 20:21 on 30 Aug, mid-session, and not filled until the next
day. Every number in `compare_models.py` and `eval_model.py` defaults to the jay set,
so the first reading added data from one speaker and scored it entirely on another,
and reported "no measurable difference" as though that meant something.

Keeping the jay table below because it is still the correct jay result, and because
the failure mode is worth naming: **a holdout that is missing a speaker does not
return a worse number, it returns a confident irrelevant one.**

| | `9a938fb` | `2213187` |
|---|---:|---:|
| plain / run-on at threshold 0.5 | 97% / **74%** | 97% / **82%** |
| plain / run-on at 6/32 FA | **100** / 82 | 97 / 82 |
| plain / run-on at 8/32 FA | **100** / **91** | 97 / 84 |
| plain / run-on at 10/32 FA | 100 / 96 | 100 / 96 |
| alignment peak / band | 160 ms / 80-240 ms | 160 ms / **80-320 ms** |
| `extend` + `hey_other` | 6/32 | 6/32 |
| ordinary negatives | 0/68 | 0/68 |
| median latency from speech end | ~50 ms | 115 ms |

Identical at matched precision. **That is the expected result of adding another
speaker's clips and scoring on jay** - roughly neutral, which is what it read. It is
not evidence that the extra ryan data did nothing; it is evidence that this test
cannot answer the question. The one thing it does establish is that ryan's clips did
not *cost* anything on jay, and that the negatives and alignment are unchanged.

The 8-point run-on gap at threshold 0.5 collapsing to nothing at matched precision is
still worth logging as **rule 2 again** - third time measured (run 10 vs `41c5cbc`,
`9a938fb` vs `41c5cbc`, now this).

Gates, for the record: `extend` 6/32 (19%) fails the 6% gate, as every model here
does; plain 34/35 fails the 98% gate by one clip (`hey_seeree_0028`, scored 0.120 -
it also reads as the weakest clip for `9a938fb`). Both are the known standing
failures, not regressions.

### The ryan holdout arrived, and it is the worst result in this file

Scored 31 Aug on ryan clips **no model has seen**: 25 plain (6 from
`my_real_samples_holdout/ryan`, plus 19 recorded 31 Aug that have since been added to
training - so this exact comparison is valid for these two models only) and 14 run-on
from `my_real_samples_holdout/ryan_runon`. Checked for leakage first: zero
byte-identical clips shared with `my_real_samples/`.

| at matched adv FA | `2213187` plain/run-on | `9a938fb` plain/run-on |
|---|---:|---:|
| 6/32 | 28 / 79 | 24 / 79 |
| 8/32 | 40 / 79 | **64** / 79 |
| 10/32 | 56 / 86 | **72** / 86 |
| 12/32 | 64 / 86 | **80** / 86 |

**Ryan plain reads 24% where jay reads 97%, on the same model at the same threshold.**
Even at 10/32 false accepts - already past the operating point anyone would ship - it
is 56-72%. Ryan run-on is much healthier at 79-86%, so the failure is specific to the
phrase spoken alone.

The extra 17 clips did not fix it, and `2213187` is if anything *behind* `9a938fb`
here. n=25 is small, but not small enough to explain a 70-point gap.

### Why: ryan is four years old, and nothing in the corpus sounds like him

This is the explanation, and it was sitting outside the data the whole time. **Ryan is
a 4-year-old child.** Every other voice the model has ever seen is an adult:

* the ~30 distinct Kokoro English voices are all adult, and they are the overwhelming
  majority of the positive set
* jay is an adult male, and 160 of the 227 real clips
* openwakeword's `PitchShift` is **-3 to +3 semitones at p=0.25**
  (`openwakeword/data.py:628`)

Measured F0, autocorrelation over voiced frames, median per clip:

| | median F0 | vs ryan |
|---|---:|---:|
| ryan (age 4), n=78 | **291 Hz** (p10 254, p90 378) | - |
| jen, n=8 | 269 Hz | -1.3 st |
| jay, n=160 | 153 Hz | -11.1 st |
| Kokoro `am_adam` | 132 Hz | **-13.6 st** |
| Kokoro `af_bella` | 227 Hz | -4.3 st |

The augmentation covers **±3 semitones of a 13.6-semitone gap, a quarter of the time**.
So the model was asked to fit a speaker whose fundamental sits outside the range of
almost everything else in the corpus, with 59 clips against thousands, and it
declined. 34% on his own training clips is what that looks like.

**Correction to an earlier draft of this section:** it claimed `PitchShift` is
formant-preserving and therefore the wrong tool. That is wrong. `torch_pitch_shift`
(what `torch_audiomentations.PitchShift` calls) is a phase-vocoder `TimeStretch`
followed by `Resample` - `torch_pitch_shift/main.py:156-168` - which is exactly the
resample trick with the duration change undone. It moves formants with pitch, same as
resampling. **The binding constraint is the range and probability, not the mechanism.**

The real limitation is subtler: pitch and formants do not scale together between an
adult and a child. Ryan's F0 is 2.20x `am_adam`'s, but a 4-year-old's vocal tract is
only ~1.4-1.5x shorter, so formants should move ~1.5x. A single resample ratio cannot
satisfy both - R=2.2 gets the pitch right and overshoots the formants (chipmunk), R=1.5
gets the formants right and leaves the pitch 6 semitones low. Also unreproduced by any
resampling: a 4-year-old's less precise articulation and much larger token-to-token
variability.

### It is not a recording problem, and not a speaking-rate problem

Both were tested before the explanation arrived, and both are dead - worth keeping,
because they rule out the boring causes and leave the speaker gap as the whole story:

| | jay train | ryan train | jay holdout | ryan holdout |
|---|---:|---:|---:|---:|
| median peak | 0.09 | 0.08 | 0.17 | 0.20 |
| median RMS | 0.0139 | 0.0175 | 0.0285 | 0.0284 |
| median SNR | 24.7 dB | 24.8 dB | 26.5 dB | 26.6 dB |
| clipped clips | 0 | 0 | 0 | 0 |

Level, noise floor and clipping are indistinguishable. `check_alignment.py` on the
ryan holdout gives lead 20 ms, trail 0 ms, speech-end-to-window-end 0 ms - the same
placement as the trained ryan clips, so the segmenter is not cutting them badly.

Speaking rate looked promising - ryan's *training* clips have an 880 ms median against
jay's 715 ms, and real clips get background noise, RIR, EQ, pitch shift and gain from
openwakeword's augmentation but **never a time-stretch**, so delivery rate is the one
axis never varied for real speech. It does not hold up. Ryan's unseen clips have a
720 ms median, matching jay's 715 ms, and score badly in *every* duration bucket:

| clip duration | jay train det@0.5 | ryan unseen det@0.5 |
|---|---:|---:|
| 0-650 ms | 72% (n=36) | 12% (n=8) |
| 650-800 ms | 96% (n=80) | 38% (n=8) |
| 800-1000 ms | 88% (n=16) | 25% (n=4) |
| 1000+ ms | 82% (n=28) | 20% (n=5) |

(An earlier version of this test streamed the raw clips instead of padding them into
the 2 s window the way `eval_model.py` does, and reported jay at 18%. Any scorer that
does not reproduce `eval_model`'s 88% on jay-train is measuring its own padding.)

### The measurement that makes it unambiguous

`2213187` detects **34% (20/59) of the ryan clips it trained on**, against **88%
(140/160) of jay's**. Not a generalisation gap - it never fit the training data for
this speaker, which is exactly the signature of a voice outside the corpus
distribution rather than one merely under-represented in it.

That reframes the whole file. Every headline number above - "99% plain, 95% run-on" -
is **jay-only, and adult-only**. Nobody noticed because the holdout had no second
speaker in it, and no run to date has had a child voice to hold out.

### What to do next

The problem is now well-posed: this is a child-speech coverage problem, not a
hyperparameter problem. In rough order of expected value:

1. **Vocal-tract-length perturbation across the whole positive corpus.** Resample by
   R, then restore the duration - equivalently, widen `PitchShift` well past +3
   semitones, since that is the same operation. The point is the *range*: the corpus
   needs positives sitting at 250-350 Hz, and today it has almost none. Applies to
   every Kokoro clip, not just the 227 real ones, so it is the only lever that
   reaches the whole corpus.

   Audible demo in `vtlp_demo/` (gitignored, regenerate from the run 12 commands).
   `am_adam` at R=1.15/1.30/1.50/1.80/2.20 and `af_bella` at R=1.15/1.28/1.45, each
   as `_a_faster` (resample only, duration shrinks) and `_b_samelength` (duration
   restored with `atempo` - the variant an augmentation should use), plus
   `zz_REAL_ryan_age4.wav` and `zz_REAL_jay_adult.wav` to compare against:

   | clip | F0 | vs ryan |
   |---|---:|---:|
   | `am_adam_R1.00_original` | 132 Hz | -13.6 st |
   | `am_adam_R1.50_b_samelength` | 205 Hz | -6.0 st |
   | `am_adam_R2.20_b_samelength` | 296 Hz | +0.3 st |
   | `af_bella_R1.28_b_samelength` | 286 Hz | -0.3 st |
   | `zz_REAL_ryan_age4` | 291 Hz | 0 |

   **Listening test (jay, 31 Aug) - this is the design constraint:**

   * `af_bella_R1.28_b_samelength` is **the closest thing to ryan** in the set.
   * male voices "sound like teenagers up to R1.30 and useless above that (chipmunk)".

   So the ratio must be **conditioned on the voice's sex**, not applied globally. One
   global range either leaves the female voices short of ryan or drives the male
   voices into artefact:

   | voice class | usable R | resulting F0 | what it covers |
   |---|---|---|---|
   | `af_`/`bf_` (~227 Hz) | **1.20-1.35** | 272-306 Hz | ryan directly |
   | `am_`/`bm_` (~132 Hz) | **1.15-1.30** | 152-172 Hz | the jay-to-ryan gap, not ryan |

   Male voices are worth stretching anyway - not to reach a 4-year-old, which they
   cannot do cleanly, but to stop 132-153 Hz being the only place the corpus has any
   density. Above R=1.30 they are artefact and would teach the model chipmunk, which
   is the run 5 failure mode (training on a cue that is not the target).
2. **Far more clips of ryan**, and accept that a 4-year-old needs more of them than an
   adult does - delivery varies much more token to token at that age, so the corpus
   has to cover that variance rather than a single canonical rendition.
3. **Per-speaker `--real-copies` weighting.** The copy count is global today
   (`train.py:744`), so 160 jay clips outweigh 59 ryan ones 2.7:1 in a corpus where
   ryan is the speaker at risk. This is a small change and worth doing alongside 1.
4. **Keep a per-speaker holdout permanently** and score every run on each speaker
   separately, never pooled. Pooling would have hidden this indefinitely.
5. The `extend` idea (real near-miss recordings) drops below all of this in priority.

**A separate model for ryan is also legitimate** if 1-3 do not close the gap.
Detecting one wake word across an adult male and a 4-year-old with one 100k-parameter
model is a genuinely harder problem than any run in this file was set up to solve, and
two models with an OR gate costs one extra inference per frame.

**Ship candidate is unchanged: `9a938fb`** - but now for a weaker reason than before.
On jay the two are indistinguishable, `9a938fb` has the tighter firing band, and
there is no measurement either way on ryan. If the point of the extra clips was to
improve ryan, `2213187` is the better bet on priors and simply has not been scored.
`2213187` was not converted to tflite pending that.

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

## Run 9: `2994f49` — best model so far; the gain is real, the mechanism is not what was predicted

| | run 7 | run 9 |
|---|---:|---:|
| held-out plain (35) | 89% | **97%** |
| held-out run-on (57) | 56% | 53% |
| `extend` + `hey_other` FA | 7/32 | 7/32 |
| synthetic `cmd_run` | 36/36 | 36/36 |
| synthetic speed 1.40x / 1.60x | 3/6, 2/6 | **3/6, 2/6** |

**The direct target did not move.** Medians at those speeds went 0.465 -> 0.468 and
0.005 -> 0.046. `PLAIN_SPEEDS = (0.7, 1.6)` and `RUNON_SPEEDS` up to 1.6 were verified
present in the commit that produced this model, so training did cover the range.

**The held-out gain IS attributable to the speed change.** The training server's real
corpus was unchanged at 160 jay + 35 ryan - jen's clips and ryan's extra 7 exist only
on the recording machine and were never copied over - so run 9 differs from run 7 in
the speed range alone. +8 points of held-out plain detection came from that one change.

**But not by the predicted mechanism.** A fix for fast speech should concentrate in the
short-duration bins; run 7 was already 16/16 on 400-500 ms clips, and run 9's extra
detections are spread across 0-400, 600-800 and 800+. Combined with the synthetic
speed sweep not moving at all, the likelier reading is that wider speed variation acts
as general augmentation - more varied positives, better generalisation overall -
rather than specifically teaching fast delivery. Worth keeping either way, but the
next speed change should not be expected to help fast clips in particular.

**The synthetic speed sweep is a weak instrument and should not be treated as a gate.**
Its failures are voice-specific and consistent across speeds - `af_bella`, `af_heart`
and `af_sarah` fail at both 1.40x and 1.60x while `af_nova` and `af_sky` pass both. At
6 voices per point, "3/6" is three particular voices being hard, not a speed
threshold. Widening it to all 42 voices would make it worth reading.

**Keep it.** False accepts did not worsen - the risk flagged when staging it, that
shorter positives resemble the first syllable of "hey serious", did not materialise -
and it is the largest single-variable gain in held-out plain detection so far.

**Run 9 is the ship candidate**, ahead of run 7 on plain by 8 points for 3 points of
run-on.

---

## Batched TTS: `9a938fb` — 4.85x faster generation, no measurable quality cost

Kokoro renders several utterances per request, split apart on the server's word
timestamps (`--tts-batch 16`). A short request is ~75% fixed overhead - ~119 ms fixed
plus ~42 ms per second of audio - so batching a sub-second phrase amortises most of
it. Positive generation went **17:14 -> 3:33**.

This changed the CORPUS, unlike the GPU-resident feature patch: plain speeds now come
from a 19-value grid instead of a continuous draw (every clip in a batch must share
one voice and speed), phrases carry mid-sequence prosody, and levels are ~12% lower.
So it needed validating rather than assuming.

| | non-batched (`41c5cbc`) | batched (`9a938fb`) |
|---|---:|---:|
| alignment peak | 160 ms | **160 ms** |
| firing band | 40-320 ms | **80-240 ms** |
| `extend` + `hey_other` | 4/32 | 6/32 |
| ordinary negatives | 0/68 | **0/68** |
| held-out plain, 6/32 FA | 97% | **100%** |
| held-out run-on, 6/32 FA | 75% | **82%** |
| held-out run-on, 8/32 FA | 95% | 91% |

Comparable throughout, with the differences inside the +/-10 point noise band that
replicates established. **The alignment band is the reassuring part** - 80-240 ms is
the tightest of any recent model, so mid-sequence prosody did not move where the
phrase sits in the window, which was the specific risk.

Keep batching on.

### `9a938fb` IS THE SHIP CANDIDATE

Not because it beats the others - it does not, measurably; at matched precision it
sits inside the noise band with `eea1c56` and `41c5cbc`. It is the candidate because
among three statistically indistinguishable models it has the cleanest supporting
evidence:

* **tightest alignment band**, 80-240 ms, peak 160 ms - the least latency headroom
  wasted, and furthest from both failure modes (a band reaching 0 ms means firing
  before the word ends; a peak past 400 ms means trailing silence in training)
* **every ordinary negative category at 0** - general conversation, bare commands,
  other assistants, running speech, 0/68 in total
* **best at 6/32 false accepts** (100% plain / 82% run-on), which is nearer a
  realistic operating point than the looser matched points
* produced by the current pipeline end to end, so it is the one that is actually
  reproducible from a commit

Deploy notes:

* **Do not deploy at threshold 0.5.** Tune it on a false-accept budget - the same
  model reads 74% run-on at 0.5 and 82-91% lower down. Latency also reads 123 ms at
  0.5, over the 120 ms gate, purely as an artefact of the operating point.
* **Validate the chosen threshold against a long recording of the deployment room**
  before going below ~0.1. The negative corpus is a few minutes of audio; a wake word
  runs continuously.
* Convert with `onnx2tflite.py`, which verifies the conversion numerically - a
  wrong-axis tflite loads cleanly and detects nothing.

The caveat that applies to all three: they are separated by less than the measurement
can resolve. 35 plain and 57 run-on clips from one speaker in one session means a
single clip is 3 and 1.8 points. A second held-out session is worth more than another
training run.

**Two pipeline failures found on the way here, both silent:**

* `train.py` reported "TRAINING COMPLETE!" after a CUDA OOM killed training at 75%,
  pointing at the PREVIOUS run's model - `setup_training_dirs` clears the working
  directory but not `my_custom_model/<name>.onnx`. That stale model was evaluated
  twice before identical checksums across six matched-precision points gave it away;
  at threshold 0.5 alone it looked like a plausible new result. Both `train.py` and
  `run-training.sh` now verify the model was actually rewritten, and treat freshness
  rather than the exit code as ground truth - openwakeword exits 1 on the known
  tflite conversion failure *after* saving a good `.onnx`.
* The OOM itself was openwakeword moving the whole false-positive validation set to
  the GPU in one 2.76 GiB allocation, on top of 16.6 GiB of resident features. Now
  batched at 4096 rows, and `run-training.sh` stops the Kokoro containers (~2.4 GiB
  of CUDA context) before training starts.

---

## Replicates: what varies run to run is the THRESHOLD, not the model

`41c5cbc` repeats run 10's configuration exactly - 50k steps, `--real-copies 10`,
`max_negative_weight 2000` - differing only by the GPU-resident features patch, which
changes no training dynamics. So it is a replicate, and with the two 100k runs there
are now two samples at each of two configurations.

At threshold 0.5 the replicates look 10 points apart:

| | 50k #1 (run 10) | 50k #2 (41c5cbc) |
|---|---:|---:|
| held-out plain | 91% | 97% |
| held-out run-on | 77% | 67% |

At MATCHED false-accept counts they are identical:

| FA 8/32 | plain | run-on |
|---|---:|---:|
| 50k #1 | 97% | **95%** |
| 50k #2 | 100% | **95%** |
| 100k #1 | 97% | 77% |
| 100k #2 | 100% | 86% |

| config | plain (mean, spread) | run-on (mean, spread) |
|---|---|---|
| **50k** | 99% (3) | **95% (0)** |
| 100k | 99% (3) | 82% (9) |

**Two conclusions, one methodological and more important than the other.**

**50k beats 100k for run-on detection**, replicated, with zero spread between the 50k
runs. The run 11 verdict was right; the VRAM run that appeared to overturn it sat
inside 100k's own wide spread. Keep `--training-steps 50000`.

**What varies between runs is where the score distribution sits, not how well the
model separates the classes.** Two 50k models reading 77% and 67% at threshold 0.5
both reach 95% at 8/32 false accepts. The "run-to-run variance" diagnosed earlier is
largely threshold placement.

That means **comparing models at a fixed threshold has been misleading throughout
this document**, and any conclusion drawn from a few points of difference at 0.5
should be re-checked at matched precision before it is trusted. It also means the
detection threshold is worth more than any training change measured here: 41c5cbc
goes from 67% to 95% run-on by moving it, with no retrain.

Thresholds for `41c5cbc`:

| thr | plain | run-on | `extend`+`hey_other` | other negatives |
|---|---|---|---|---|
| 0.50 | 97% | 67% | 4/32 | 0/68 |
| 0.10 | 97% | 75% | 7/32 | 1/68 |
| 0.05 | 97% | 81% | 8/32 | 3/68 |
| 0.01 | 100% | 95% | 8/32 | 4/68 |

**Do not deploy at 0.01 on this evidence.** The negative corpus is ~100 clips, a few
minutes of audio; openwakeword's own tuning targets false-positives-per-HOUR against
11.3 hours. The `other negatives` column going 0 -> 4 across that range is the one to
watch, since those are ordinary speech. Validate against a long recording from the
deployment room before choosing anything below ~0.1.

---

## Run 11: `77aa984` — 100k steps looked better and was worse

At threshold 0.5 it is the best model yet. At matched precision it is the worst of
the recent runs.

| threshold 0.5 | run 10 | run 11 |
|---|---:|---:|
| held-out plain | 91% | **100%** |
| held-out run-on | 77% | **84%** |
| `extend` + `hey_other` FA | 6/32 | **10/32** |
| `extend` median score | 0.017 | **0.292** |

Held-out detection at MATCHED false-accept counts, threshold tuned per model:

| FA | run 10 | run 11 |
|---|---|---|
| 4/32 | **91% / 68%** | 80% / 58% |
| 6/32 | **94% / 81%** | 94% / 68% |
| 8/32 | **97% / 95%** | 97% / 77% |
| 10/32 | **100% / 95%** | 100% / 84% |

Run 10 is better or equal at every point, and much better on run-ons. Run 11's
apparent gain was entirely an operating-point shift: doubling `--training-steps`
halves the rate of the negative-weight ramp (`np.linspace(1, max_negative_weight,
steps)`), so the model is penalised less for false positives throughout training. The
whole score distribution moved up - the `extend` median went 0.017 -> 0.292.

**Reverted to 50k.** This is the second time a change looked good at a fixed
threshold and vanished under matched-precision comparison; run 8 was the first, in
the opposite direction. **Any future change that moves detection and false accepts
the same way should be checked this way before it is believed.**

### The bigger finding: threshold 0.5 is a poor operating point

Run 10 across thresholds, on held-out real recordings:

| thr | plain | run-on | `extend`+`hey_other` | all other negatives |
|---|---|---|---|---|
| **0.50** | 91% | 77% | 6/32 | 2/68 |
| 0.25 | 94% | 84% | 7/32 | 3/68 |
| 0.15 | 94% | 86% | 7/32 | 3/68 |
| **0.05** | **97%** | **91%** | 7/32 | 3/68 |

Lowering the threshold to 0.05 buys +6 points plain and +14 points run-on for ONE
extra adversarial false accept - more than run 11 gained by training twice as long,
with no retrain at all. Run 10's score distribution is strongly bimodal: real
positives score high, negatives score near zero, and very little sits between.

**Validate before deploying below ~0.1.** The negative corpus is ~100 clips, only a
few minutes of audio. A wake word runs continuously, and openwakeword's own tuning
targets false-positives-per-HOUR against an 11.3-hour validation set. A threshold
that costs one extra false accept across five minutes of adversarial clips may cost
many per hour on real background audio. Test on a long recording from the room the
satellite lives in before committing.

---

## Run 11 (setup): `--training-steps` 50k -> 100k

Never tested. Distinct feature vectors went up ~4.5x across runs 4-10 (samples_per_voice
200 -> 300, augmentation_rounds 1 -> 3, real_copies 3 -> 10) while steps stayed at
50,000, so each vector is revisited far less than it used to be. This is the cheapest
untried lever: ~9 extra minutes on a run that now takes ~35.

**It is not purely "train longer".** openwakeword derives `warmup_steps` (steps/5),
`hold_steps` (steps/3) and the negative-weight ramp `np.linspace(1, max_negative_weight,
steps)` from this value, and runs two further sequences at steps/10 each. Doubling it
therefore also **halves the rate at which the negative weight climbs** - at any given
step the model is penalised less for false positives than it was before. Run 8 showed
that weight schedule moves detection and false accepts in opposite directions, so both
should be expected to shift.

| | run 10 | run 11 |
|---|---:|---|
| held-out plain | 91% | **rises** -> was under-trained |
| held-out run-on | 77% | **rises** -> same |
| `extend` + `hey_other` FA | 6/32 | may *worsen* - slower weight ramp |
| trained-set plain | 98% | if this rises while held-out is flat, it is memorising |

If everything is flat, 50k was already enough and this is settled cheaply. If detection
rises while false accepts worsen, that is the weight-schedule side effect rather than
better training, and the two can be separated by re-running at 100k with
`--max-negative-weight 4000` to restore the original ramp rate.

Corpus unchanged from run 10 (160 jay + 35 ryan on the training server; jen's 8 and
ryan's newest 7 are still only on the recording machine).

---

## Run 10: `eea1c56` — real-sample weighting is the biggest lever found

`--real-copies 3 -> 10`, single variable. Real goes from 4.4% to 13.4% of the positive
class, using the same 195 clips.

| | run 7 | run 9 | run 10 |
|---|---:|---:|---:|
| held-out plain (35) | 89% | **97%** | 91% |
| **held-out run-on (57)** | 56% | 53% | **77%** |
| `extend` + `hey_other` FA | 7/32 | 7/32 | **6/32** |
| `running` FA | 1/12 | 0/12 | 2/12 |
| median latency | 83 ms | 91 ms | **49 ms** |
| trained-set plain (56) | 98% | 95% | 98% |

**Run-on rose 14 clips** (30/57 -> 44/57), far outside noise. Plain fell 2 clips on a
35-clip set, which is not. So the result is unambiguous: **the synthetic/real balance
was the constraint**, and it was reachable by reweighting alone - no new recordings.

This was the outcome bet against when staging it. Duplication is weighting, not
augmentation, so overfitting to the 195 clips was the expected failure. The trained-set
number did rise (95% -> 98%), which is that signature, but held-out run-on rose far
more, so generalisation won.

**Why run-on and not plain** is worth noting: the real recordings are all plain wake
words - there are no real run-ons in training. Weighting them did not add run-on
examples, yet run-on improved most. The likeliest reading is that real speech teaches
acoustics (room, mic, delivery) that synthetic speech does not, and that matters most
in the hardest case rather than the easiest.

**Costs:** `running` false accepts went 0/12 -> 2/12, worst 0.951 - more real positives
makes the model more responsive to real-ish speech generally. Worth watching if it
grows.

**Run 10 is the ship candidate:** best run-on by 21 points, plain within noise of best,
best false accepts since run 4, second-best latency.

**What this implies for the next lever.** Reweighting 195 clips bought 24 points of
run-on. Adding real *variety* - jen's clips and ryan's newest are still only on the
recording machine, and neither has a held-out set - should do at least as well and
carries no overfitting risk. That is now clearly the highest-value work, ahead of any
further parameter tuning. Pushing `--real-copies` higher (20x) is the cheaper probe but
runs into the overfitting the trained-set number is already hinting at.

---

## Run 10 setup: weight real recordings 3x -> 10x

A cheap probe of the dominant hypothesis for the held-out gap. Positives are 95.6%
synthetic - 12,600 Kokoro clips against 195 real from 2 speakers on the training
server - and every measurement this session where synthetic and real disagreed, real
was worse:

* held-out plain 97% against ~96% on the trained set (run 9 closed this)
* held-out run-on 56% against 100% on the synthetic `cmd_run` sweep
* run 2 scored 83% synthetic `cmd_run` while detecting 3/57 real run-ons

`--real-copies 3 -> 10` moves real from 4.4% to 13.4% of the positive class (195 clips
on the training server, not the 210 on the recording machine - jen and ryan's newest
have not been copied across), with no recording effort. Batch class balance is unaffected (`batch_n_per_class` fixes that),
so this only changes how often a real clip is drawn *within* the positive class.

**This is a probe, not a fix.** Duplication is weighting, not augmentation - the same
195 clips repeated - so it buys emphasis without variety and risks overfitting to
those specific recordings. The held-out set is what will show that.

| | run 9 | run 10 |
|---|---|---|
| held-out plain | baseline | **rises** -> balance is the constraint |
| held-out run-on | baseline | **rises** -> same |
| trained-set plain | ~96% | if this rises while held-out falls, it is overfitting |
| `extend` + `hey_other` FA | baseline | watch: fewer distinct positives may cost precision |

The interesting failure is held-out flat while trained-set climbs. That would say the
synthetic/real ratio is not the constraint, the *variety* of real speech is - and the
answer is more speakers and more sessions, not more weight on the clips already held.

Keep the corpus fixed for this run: no new recordings between run 9 and run 10, or
the comparison is lost.

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

**Single-variable after all.** The recording machine gained ryan clips (35 -> 42) and
a new speaker jen (8 clips) around this time, but they were never copied to the
training server, so run 9 trained on the same 160 jay + 35 ryan as runs 7 and 8.

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
