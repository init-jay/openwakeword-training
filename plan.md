# Plan: add microWakeWord training alongside openWakeWord

Target: [OHF-Voice/micro-wake-word](https://github.com/OHF-Voice/micro-wake-word), the
wake-word runtime ESPHome uses on ESP32. Output is an int8-quantised streaming
`.tflite` plus a JSON manifest, not an ONNX.

Phases 1 and 2 are BUILT and running; phase 3 is not started. Sections written before
they were built are kept where the reasoning still holds and marked where it did not -
several confident claims here were wrong, and which ones is the useful part.

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

One repo, two trainers behind a shared corpus layer. As built:

    corpus/                  engine-agnostic, extracted from train.py
      augment.py             trimming, child-range copies
      negatives.py           the negative wordlist
      positives.py           phrase texts + speed grid (tuned; shared so they cannot drift)
      piper.py               Wyoming generation, voice audit lists, F0 sex map
      real.py                real recordings into a corpus
    train.py                 openWakeWord trainer (unchanged behaviour)
    mww/
      corpus.py              WAVs -> my_custom_model/<ww>/mww/{positives,negatives}
      features.py            -> .../mww/features/<set>/<split>/*_mmap
      config.py              -> the training YAML
      train.py               -> quantized streaming tflite
      manifest.py            NOT WRITTEN - the ESPHome JSON
    eval/                    NOT WRITTEN - phase 3

Corpora are siblings under `my_custom_model/<wake_word>/{oww,mww}/`, so neither
trainer reaches into the other's. That nesting also fixed a bug: `setup_training_dirs`
rmtree'd `my_custom_model/<wake_word>/`, which is where `run-training.sh` archives the
commit-tagged models - every run deleted the archive of every previous run.
`patches/configurable-corpus-dir.py` makes openWakeWord read the corpus location from
its config, since upstream derives it from the same value that names the model.

`1_datagen/`, `2_train/`, `3_eval/` still exist and are still empty - three
directories claiming a structure nothing uses.

**Do not refactor `train.py` before phase 2.** Its behaviour is the baseline that
sixteen runs of `tuning.md` are calibrated against; a refactor that quietly changes
corpus generation invalidates the notebook. Extract only when there is a second
consumer to prove the extraction against, and re-run one config to confirm the numbers
land in the same band.

## Phase 0 - one throwaway model — SKIPPED

Never run. Phase 2 was built directly instead, and its questions got answered the
expensive way, mid-build:

| question | answer |
|---|---|
| does `pymicro-features` build in the image? | yes on linux/amd64, py3.11 - no fork needed |
| RaggedMmap layout; can custom WAV dirs be fed in? | `<dir>/<split>/**/*_mmap`; yes via `Clips`, but **training only** - see phase 2 |
| how large are the negative sets? | 5.7 GB across four archives |
| does ESPHome accept a hand-written manifest? | still unknown - the emitter is unwritten |

**The skip cost real time.** Six distinct failures surfaced during phase 2 that a
half-day throwaway run would have surfaced first, and one of them - `type: clips`
being training-only - had already been written into this plan as a finding that
"deletes most of what this phase was scoped to build". Reading the source is not the
same as running it, which is what this phase existed to say.

## Phase 1 - shared corpus layer (2-3 days)

**Status: DONE as scoped — extraction, Piper generator, gate all complete. Two caveats
below, and note this phase ran out of order.**

The gate ran as `7075c91` and was accepted: jay run-on read 9 points below the
four-run cluster, ryan identical, jen equal at 10/32, with plain detection better.
Judged close enough on the strength of the code equivalence rather than replicated -
see the gate section in tuning.md, including why no seed makes that question
expensive to answer properly.

Then the Piper generator earned its keep immediately and outside this plan's scope:
run 17 (`d1bb9f4`) put 82 audited Piper voices into the openWakeWord corpus and
measured the largest gain since run 10 - jay run-on 75% -> 95% at 8/32, ryan plain
83% -> 100%, adversarial false accepts down. The shared corpus layer is therefore
proven useful to the trainer that already existed, independent of microWakeWord.

**Caveat 1 (RESOLVED in phase 2).** `corpus/` output does feed microWakeWord: its
`Clips(input_directory, file_pattern)` reads the same directories of 16 kHz WAVs. The
qualification is that clips alone serve training only, so `mww/features.py` converts
them to RaggedMmap for validation and testing.

**Caveat 2: this phase ran BEFORE phase 0 and phase 3's interface, inverting the
order this plan argued for.** The reasoning for that order was that mWW's data layout
should constrain the corpus layer's design, and that models are cheap while
trustworthy measurement is not. Nothing has gone wrong yet, but the risk it named is
still open: `corpus/` may need rework once mWW is actually attempted.

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

`--use-cuda` was measured on the training server and REMOVED: CUDA runs but makes
Piper 2.5x slower (17.45 clips/s CPU against 7.00), because ORT splits the VITS graph
and pays 28 host<->device copies per inference. One CPU instance, no second instance
(0.60x). See docker-compose.yml for the numbers.

Extract the frontend-agnostic functions from `train.py` into `corpus/`, add a Piper
generator beside `KokoroPool`, and make both trainers consume WAV directories.

Kokoro audio is just WAVs, so **both engines can feed both trainers** - the mWW corpus
does not have to be Piper-only, and voice diversity was measurable here (embedding
distance 0.70 between voices vs 0.09-0.44 across prosody variants). Worth testing;
upstream does not do it.

Gate: re-run one openWakeWord config through the extracted layer and confirm it lands
inside the noise band on jay/ryan/jen. If it does not, the extraction changed something.

## Phase 2 - mWW training backend

**Status: the pipeline runs end to end.** Corpus -> features -> config -> training ->
quantized streaming tflite. Written from the source rather than the README, and twice
corrected by running it - both corrections are below, because the wrong version was
confidently written down first.

### The pipeline, as built

    mww/corpus.py    WAVs      my_custom_model/<ww>/mww/{positives,negatives}
    mww/features.py  RaggedMmap  .../mww/features/{positives,negatives}/<split>/*_mmap
    mww/config.py    YAML      mww_models/<ww>/<tag>.yaml
    mww/train.py     model     mww_models/<ww>/<tag>/tflite_stream_state_internal_quant/

`Dockerfile.mww` + an `mww` compose service; `setup-mww-data.sh` for the ambient sets.

### Correction 1: `type: clips` is TRAINING-ONLY, so the mmap step is required

The first reading of the source found that a feature set can be `type: clips`, which
reads a directory of WAVs and generates spectrograms on the fly, and concluded that
"there is no mmap step to build - that deletes most of what this phase was scoped to
build." **That was wrong**, and it was wrong because the check stopped at "does this
type exist" without asking which MODES it serves:

    # data.py, ClipsHandlerWrapperGenerator
    def get_mode_size(self, mode):
        if mode == "training":
            return len(self.spectrogram_generation.clips.clips)
        else:
            return 0

Validation and testing get nothing from a clips set. The symptom is a shape error at
the first validation step, from whatever the ambient sets yield instead.

A clips training set would also LEAK: it is built with
`spectrogram_generator(random=True)`, which draws from `Clips.clips` - every clip in
the directory, ignoring the train/validation/test split - so training would sample the
clips validation is scored on. That would not have crashed. It would have inflated
validation accuracy quietly, which is worse.

So `mww/features.py` writes all three splits to RaggedMmap up front, with
`slide_frames` 10/10/1 per upstream, and every feature set in the config is
`type: mmap`. Augmentation happens there, once, rather than in the config.

### Correction 2: only the `_eval` archives can drive model selection

The four Hugging Face sets are not interchangeable, and the names do not say so:

| set | splits | role |
|---|---|---|
| `speech`, `no_speech`, `dinner_party` | `training` | ambient negatives to train on |
| `dinner_party_eval` | `validation_ambient`, `testing_ambient` | **model selection** |

`maximization_metric` is `average_viable_recall`, computed from false accepts per hour
on ambient audio. Without an `*_ambient` split it reads **0.000 at every step**, the
best checkpoint never improves on anything, and the exported model is whichever
happened to be current - while accuracy, recall, precision and AUC all still look
excellent. Measured: 96.4% accuracy, 97.6% recall, AUC 0.978, and a selection metric
of exactly zero.

Pass all four to `--ambient`. `setup-mww-data.sh` now prints which splits each set
provides and warns when none supplies the evaluation ones.

### What held up from the first reading

- **The augmentation corpora are the ones already downloaded** - `Augmentation` takes
  `impulse_paths` and `background_paths`, which is `data/mit_rirs`,
  `data/audioset_16k`, `data/fma` from `setup-data.sh`. Nothing to re-fetch.
- **Derived config values are computed by the trainer** - `spectrogram_length`,
  `training_input_shape`, `stride` come from `model_train_eval.py:60-93`. The
  generator writes only authored keys.
- **mWW cannot share the trainer image** - it requires `numpy>=2.0` against this
  repo's `numpy<2`. A hard incompatibility, and a good separation anyway.

### What running it actually cost, and why

Six failures, none of which the README would have predicted, and all but one silent
or misleading rather than a clean error:

1. `Unknown model type: None` - the architecture is an argparse SUBCOMMAND
   (`mixednet`), not a config key, and its flags carry the notebook's settings.
2. `all input lists have to be the same length` - upstream's own argparse defaults
   are inconsistent (`residual_connection` has 5 entries against 4 filters).
3. `ImportError: ... install 'torchcodec'` - mWW's `setup.py` leaves `datasets`
   unpinned and 4.x moved audio decoding to torchcodec. Pinned to `datasets[audio]<4`.
4. `TBNotInstalledError` - `tf.summary.scalar` needs tensorboard, which neither the
   TF image nor mWW's setup.py provides. Training dies after exactly one eval
   interval.
5. `model already exists in folder` - `os.makedirs(train_dir)` refuses any existing
   directory, including one holding only a config file. Runs are now tagged per
   commit, config written as a sibling.
6. The two corrections above.

Each is now a build-time assert or a refusal in `mww/train.py`, because every one of
them either produced a confusing traceback deep into a run or, worse, did not.

### Still to build

- **`mww/manifest.py`** - the ESPHome JSON. `probability_cutoff` and
  `sliding_window_size` should come from `tflite_streaming_roc.txt`, which the
  training run writes beside the model: false accepts per hour against cutoff. That
  is the same job as the threshold sweep on the openWakeWord side, and picking a
  default instead would repeat the mistake `tuning.md` opens by warning about.
- **Kokoro in the mWW corpus.** It is Piper-only today because the Kokoro client
  still lives in `train.py` rather than `corpus/`. Run 17 measured two engines
  beating one by the largest margin since run 10, so this is a known cost.
- **Run-on positives.** Their cut point needs word timestamps Wyoming does not
  expose. On the openWakeWord side run-ons took held-out run-on detection from 5% to
  the 80s - the most valuable gap here.
- **The shared corpus pool** (deferred until the first mWW model trained, which it
  now has). Render once, assemble per trainer by hardlink; the differences between
  the two corpora are all assembly, not generation.

### The negatives still need thought, not concatenation

mWW trains against large pre-generated ambient sets. This repo's adversarial
negatives are ~1100 clips - a rounding error beside them unless weighted, and
`extend` false accepts are exactly what they exist to fix (unsolved at 6-8/32 since
run 6). `sampling_weight` and `penalty_weight` are per feature set, which is a better
instrument than openWakeWord's single `max_negative_weight`. Untuned so far.

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

**Planned:** phase 0 → phase 3's `backends.py` interface → phase 1 → phase 2.

**Actual:** phase 1 → phase 2, with phase 0 skipped and phase 3 not started.

The argument for the planned order was: **define the eval interface before building
the trainer**, because models are cheap and trustworthy measurement is not, and eight
runs here were judged on contaminated numbers before anyone noticed. Building the
trainer first means the first mWW model arrives with no way to tell whether it is any
good - and the temptation is to judge it at a fixed threshold, which is the mistake
`tuning.md` opens by warning about.

That is now the situation. A mWW model trains, and nothing can score it:
`compare_models.py`, `eval_model.py` and `check_model_alignment.py` all load through
`openwakeword.model.Model`, which cannot read a streaming tflite with internal state.
The first model came out with `average_viable_recall = 0.000` at every step and
looked excellent by accuracy, recall, precision and AUC - which is exactly the failure
the ordering was meant to prevent.

Skipping phase 0 cost six avoidable failures (phase 2). Skipping phase 3's interface
has cost nothing yet only because no mWW model has needed judging. **Phase 3 is the
next thing, before any mWW tuning run.**
