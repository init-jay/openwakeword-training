# Plan: add microWakeWord training alongside openWakeWord

Target: [OHF-Voice/micro-wake-word](https://github.com/OHF-Voice/micro-wake-word), the
wake-word runtime ESPHome uses on ESP32. Output is an int8-quantised streaming
`.tflite` plus a JSON manifest, not an ONNX.

This is a plan, not a decision that has been taken. Everything below marked
**(verified)** was checked against this repo or upstream on 2026-09-01; everything
marked **(assumed)** has not been, and is where the estimate will move.

## The thing to get right first

**The valuable asset in this repo is not `train.py`. It is the ability to tell whether
a run got better.** Sixteen runs of `tuning.md` exist because that question is hard:
two runs of an identical config measured ten points apart at a fixed threshold, and
three separate conclusions were reversed by later runs.

microWakeWord's own README says the training notebook "will produce a model, but it
will most likely not be usable. Training a usable model requires a lot of
experimentation." That is this repo's last sixteen runs restated, by the upstream
authors, in advance.

So the cost is not the trainer. Porting corpus generation is days. Porting *the
measurement* is the project, and the plan is ordered accordingly.

## Where the two pipelines diverge

The seam is sharper than expected. Everything up to and including "a directory of
16 kHz mono WAVs" is frontend-agnostic and already written:

| asset | file | reusable? |
|---|---|---|
| silence trimming | `train.py:989` `trim_silence`, `:1035` `trim_directory` | yes - operates on arrays |
| child-range copies | `train.py:869-988` `time_stretch`, `vocal_tract_shift`, `add_child_range_copies` | yes |
| run-on positives | `train.py:693` `generate_runon_samples`, `RUNON_TAIL_MS` | yes |
| real-sample handling | `train.py:803` `copy_real_samples` (recursive, path-flattening) | yes |
| adversarial negatives | `generate_negatives.py` `EXTEND`/`RUNNING`/`HEY_OTHER`/... | yes, as audio |
| Piper voice audit | `audit_voices.py` + 252 renders in `voice_audit_piper/` | yes - **and mWW needs it more** |
| augmentation corpora | `setup-data.sh:36-97` | **already downloaded** (verified) |
| 17 GB ACAV100M features | `data/*.npy` | **no** - wrong frontend |
| trained `.onnx` models | `my_custom_model/` | **no** - no conversion path |
| eval harness | `eval_model.py`, `compare_models.py`, `check_model_alignment.py` | **no, as written** - see phase 3 |

Two of those deserve emphasis.

**The augmentation data is already on the training server (verified).**
`setup-data.sh` pulls MIT RIRs (`davidscripka/MIT_environmental_impulse_responses`),
AudioSet `bal_train09.tar` (`agkphysics/AudioSet`) and FMA - the same three sources,
from the same URLs, that mWW's notebook fetches. Nothing to download.

**The Piper audit is worth more to mWW than it is here.** mWW generates positives with
Piper and the notebook's only defence against g2p mispronunciation is a suggestion to
use "phonetic spellings". `audit_voices.py` measures it instead, and already handles
the harder Piper case: VITS is stochastic, so a speaker's score is a *contamination
rate* rather than a verdict (`en_US-libritts_r-medium:1383` scored 0% and then 100%
across passes). This is the one place the port starts ahead of upstream.

The divergence begins at feature extraction: openWakeWord is melspectrogram → embedding
model → 96-dim embeddings over a 2000 ms window; microWakeWord is 40 features per 10 ms
(30 ms window) into a streaming MixConv net over a 1500 ms clip.

## Layout

Keep one repo. Two trainers behind a shared corpus layer:

    corpus/                  extracted from train.py - engine-agnostic
      generate.py            Kokoro + Piper -> WAV dirs
      augment.py             trim, child-range, run-on
      negatives.py           from generate_negatives.py
    oww/                     existing pipeline, unchanged behaviour
      train.py
    mww/
      train.py               config + SpectrogramGeneration + MixedNet
      manifest.py            emit the ESPHome JSON
    eval/
      backends.py            the shared model interface (phase 3)
      eval_model.py  compare_models.py  check_alignment.py

`1_datagen/`, `2_train/`, `3_eval/` currently exist and are empty (verified). Either
adopt that naming or delete them - three empty directories are a false signal about
where things live.

**Do not refactor `train.py` before phase 2.** Its behaviour is the baseline that
sixteen runs of `tuning.md` are calibrated against; a refactor that quietly changes
corpus generation invalidates the notebook. Extract only when there is a second
consumer to prove the extraction against, and re-run one config to confirm the numbers
land in the same band.

## Phase 0 - one throwaway model (half a day)

Run mWW's `basic_training_notebook.ipynb` unmodified, with their Piper samples and
their pre-generated negative spectrograms, and get *any* `.tflite` onto a device.

The point is not the model, which will be bad. The point is that the notebook's data
layout constrains every later design decision, and reading it is not the same as
running it. Specifically, find out:

- whether `pymicro-features` builds in the trainer image, or needs the macOS fork
  (assumed: fine on Linux, untested)
- the exact `RaggedMmap` on-disk layout and whether custom WAV directories can be fed
  to `SpectrogramGeneration` without going through their Piper generator (assumed: yes)
- how large the negative spectrogram sets are on disk
- whether ESPHome accepts a hand-written manifest

Do not build anything reusable in this phase.

## Phase 1 - shared corpus layer (2-3 days)

**Status: extraction done, Piper generator working against a live server, gate not yet run.**

Done: `corpus/augment.py` (trimming, child-range copies), `corpus/real.py`,
`corpus/negatives.py`, `corpus/piper.py`. `train.py` imports them and defines none of
it any more; `Dockerfile` copies the package and fails the build if it does not import.

**This makes the trainer image mandatory to rebuild.** `train.py` now imports
`corpus`, so any image built before the package existed dies with
`ModuleNotFoundError: No module named 'corpus'` - at import, before any of the
expensive setup. `docker compose build trainer`.
`Dockerfile.piper` + the `piper`/`piper2` compose services host Piper with the GPU
available - built rather than pulled, because rhasspy/wyoming-piper is debian-slim
with CPU onnxruntime and cannot use a GPU at all.

Verified so far - all static or offline, none of it a substitute for the gate:

- every moved function and constant is **byte-identical** to its pre-move version,
  compared by AST against `git show HEAD:train.py`. The one intended exception is
  `copy_real_samples`, whose only body change is `real_samples_dir` becoming a
  parameter instead of a module global (the caller passes the same value).
- `trim_silence`, `time_stretch` and `vocal_tract_shift` produce **bit-identical
  output** to the originals across 60 real clips from `my_real_samples/` and
  `negatives_tts/`.
- `build_negative_phrases` returns an identical list for "hey seeree" (59 phrases)
  and "hey cal" (21).
- pyflakes reports no undefined names in `train.py` or the package, which is what
  catches a function body still reaching for a deleted global. (It reports one
  unused `time` import, which pre-dates this work.)

`corpus/piper.py` has now been run against a live server (the container above, CPU
mode, on an amd64 emulation): `piper_voices` enumerated 2005 (voice, speaker) pairs
across en_US/en_GB, and `piper_render` returned 16 kHz int16 at exact speed ratios -
0.7000, 1.2501, 1.6001 on a single rendering. Resampling 22050 -> 16000 produced no
clipping (peak 32717, zero samples out of int16 range) on the clip checked, though
Piper normalises to full scale so the guard clip in `piper_render` earns its place.

Worth knowing: measuring speed across SEPARATE renderings looks wrong, because VITS
samples durations per call - three calls at 1.0/1.6/0.7 gave 0.964 s/0.501 s/1.194 s,
which is not the 1.6x it looks like. Compare stretch ratios on one clip, not across
calls. Same stochasticity `audit_voices.py` documents.

Still to do: **the gate below**. Also untested: `--use-cuda` itself, which needs a
GPU host - and per the note in docker-compose.yml is a flag to measure rather than
assume.

Extract the frontend-agnostic functions from `train.py` into `corpus/`, add a Piper
generator beside `KokoroPool`, and make both trainers consume WAV directories.

Kokoro audio is just WAVs, so **both engines can feed both trainers** - the mWW corpus
does not have to be Piper-only, and voice diversity was measurable here (embedding
distance 0.70 between voices vs 0.09-0.44 across prosody variants). Worth testing;
upstream does not do it.

Gate: re-run one openWakeWord config through the extracted layer and confirm it lands
inside the noise band on jay/ryan/jen. If it does not, the extraction changed something.

## Phase 2 - mWW training backend (3-5 days)

Wire `corpus/` output through `SpectrogramGeneration` into the ragged mmap layout, plus
a YAML config generator mirroring `create_config` (`train.py:1073`), and manifest
emission.

Start from the notebook's config verbatim - `MixedNet`, pointwise filters 64x4,
mixconv kernels `[5],[7,11],[9,15],[23]`, 10000 steps, batch 128, negative class
weight 20 - and change one thing at a time, `tuning.md` style.

**The negatives need thought, not just concatenation.** mWW trains against large
pre-generated negative spectrogram sets (`dinner_party`, `speech`, `no_speech`). This
repo's adversarial negatives are ~100 clips. Dropped into a set that size they are a
rounding error, and `extend` false accepts - unsolved here at 6-8/32 across every run
since run 6 - is exactly what they exist to fix. Expect to need mWW's equivalent of
`max_negative_weight`. This is the same lesson `--real-copies 10` taught, and it was
the single biggest lever found here.

## Phase 3 - evaluation (the expensive part, 1-2 weeks)

All three eval tools instantiate `openwakeword.model.Model(wakeword_models=[path],
inference_framework=...)` - `compare_models.py:66`, `eval_model.py:228`,
`check_model_alignment.py:145` (verified). A mWW streaming tflite will not load there.

The good news: `check_model_alignment.py:134` already defines a `WakeWordModel` class
that wraps "onnx or tflite" behind one interface. **Generalise that into
`eval/backends.py` and point all three tools at it.** The contract both backends can
honour is the one `eval_model.py:111` `stream()` already assumes: feed 16 kHz PCM in
fixed chunks, get one score per chunk, reset between clips.

Everything downstream of that contract - per-category false accepts, per-speaker
scoring, matched-FA comparison, the latency measurement - is arithmetic over scores and
carries over unchanged. That is the payoff for doing it as an interface rather than a
fork.

Four traps, in the order they will bite:

1. **State leaks between clips.** The mWW model is `stream_state_internal` - it carries
   internal state across invocations. Without an explicit reset, clip N's score depends
   on clip N-1, and the corpus order silently becomes a variable. `clip_rng`
   (`eval_model.py:92`) exists because of an almost identical bug, where shared padding
   noise made a clip's result depend on how many clips preceded it. Verify the reset
   works by scoring one corpus in two different orders and requiring identical output.

2. **"Threshold" becomes two parameters.** ESPHome exposes `probability_cutoff` *and*
   `sliding_window_size` (verified). The single-threshold sweep in `compare_models.py`
   becomes a 2D surface, and rule 2 of `tuning.md` - never compare at a fixed threshold,
   always at matched false accepts - needs restating for two knobs. Simplest honest
   version: fix `sliding_window_size` per comparison, sweep the cutoff, and report the
   window size alongside every number.

3. **int8 quantisation coarsens the scores.** The current sweep goes down to 0.01 and
   the gap between 0.02 and 0.05 has been meaningful (run 16: jay run-on 95% vs 86%).
   A quantised model may not have that resolution. Check the actual distinct score
   values before designing the sweep around fine thresholds.

4. **Alignment means something different.** `check_model_alignment.py` sweeps where the
   phrase sits in a 2000 ms window; mWW's clip duration is 1500 ms and it is a streaming
   detector with a sliding window average. The latency floor is still measurable and
   still matters - it is the deployed number - but the current tool's framing does not
   transfer directly. Redesign rather than port.

The eval corpora themselves - `my_real_samples_holdout/`, `negatives_tts/` - are WAVs
and carry over untouched.

## What this plan does not do

- **No shared model format.** The two produce different artifacts for different targets.
  ONNX for the server, streaming tflite for the ESP32.
- **No merging of `tuning.md`.** Start `tuning_mww.md`. The numbers are not comparable -
  different frontend, different window, different threshold semantics - and pooling them
  would produce exactly the confident-irrelevant reading the per-speaker rule exists to
  prevent.
- **No shared training data.** Only shared *audio*. The 17 GB of features is
  openWakeWord's alone.

## Open questions

1. **Does the child-range lever transfer?** `add_child_range_copies` moved ryan from
   24% to 83%, the biggest single win in the notebook. Whether pitch/formant-shifted
   copies survive a 40-feature 10 ms frontend and int8 quantisation is unknown, and it
   is the first experiment worth running after a baseline exists.
2. **Do the real recordings matter as much?** `--real-copies 10` was the biggest lever
   here. 331 clips against mWW's much larger negative sets may need heavy weighting to
   register at all.
3. **Is `jen_runon/` recorded first?** It is already owed to the current pipeline
   (run 16). Any speaker-coverage claim about a mWW model inherits the same gap.
4. **Which device is the target?** Tensor arena size and `feature_step_size` go in the
   manifest, and the answer changes what "works" means.

## Order of work

Phase 0 → phase 3's `backends.py` interface → phase 1 → phase 2.

Deliberately not sequential: **define the eval interface before building the trainer.**
The pattern from sixteen runs is that models are cheap and trustworthy measurement is
not, and eight runs here were judged on contaminated numbers before anyone noticed.
Building the trainer first means the first mWW model arrives with no way to tell
whether it is any good - and the temptation will be to judge it at a fixed threshold,
which is the mistake `tuning.md` opens by warning against.
