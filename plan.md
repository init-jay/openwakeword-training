# Plan: add microWakeWord training alongside openWakeWord

Target: [OHF-Voice/micro-wake-word](https://github.com/OHF-Voice/micro-wake-word), the
wake-word runtime ESPHome uses on ESP32. Output is an int8-quantised streaming
`.tflite` plus a JSON manifest, not an ONNX.

Phases 1, 2 and 3 are BUILT and running; phase 0 was skipped. Sections written before
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
| silence trimming | `train/oww/train.py:989` `trim_silence`, `:1035` `trim_directory` | yes - operates on arrays |
| child-range copies | `train/oww/train.py:869-988` `time_stretch`, `vocal_tract_shift`, `add_child_range_copies` | yes |
| run-on positives | `train/oww/train.py:693` `generate_runon_samples`, `RUNON_TAIL_MS` | yes |
| real-sample handling | `train/oww/train.py:803` `copy_real_samples` (recursive, path-flattening) | yes |
| adversarial negatives | `eval/generate_negatives.py` `EXTEND`/`RUNNING`/`HEY_OTHER`/... | yes, as audio |
| Piper voice audit | `tools/audit_voices.py` + 252 renders in `voice_audit_piper/` | yes - **and mWW needs it more** |
| augmentation corpora | `scripts/setup-data.sh:36-97` | **already downloaded** (verified) |
| 17 GB ACAV100M features | `data/*.npy` | **no** - wrong frontend |
| trained `.onnx` models | `my_custom_model/` | **no** - no conversion path |
| eval harness | `eval/` (`eval_model.py`, `compare_models.py`, `check_model_alignment.py`) | **no, as written** - see phase 3 |

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

One repo, two trainers behind a shared corpus layer. **Reorganised to follow
`architecture.md`'s four pipeline steps** - one directory per step, no loose scripts at
the root:

    record/                  1 - record and check real speech (own uv env)
      record_samples.py, check_alignment.py
    train/                   2 - training run
      corpus/                engine-agnostic, extracted from train.py, shared
        augment.py           trimming, child-range copies
        negatives.py         the negative wordlist
        positives.py         phrase texts + speed grid (shared so they cannot drift)
        piper.py             Wyoming generation, voice audit lists, F0 sex map
        real.py              real recordings into a corpus
      oww/
        train.py             openWakeWord trainer (unchanged behaviour)
        onnx2tflite.py       -> the .tflite that pyopen-wakeword can read
      mww/
        corpus.py            WAVs -> my_custom_model/<ww>/mww/{positives,negatives}
        features.py          -> .../mww/features/<set>/<split>/*_mmap
        config.py            -> the training YAML
        train.py             -> quantized streaming tflite
        manifest.py          -> the ESPHome/LVA JSON
    eval/                    3 - eval model
      generate_negatives.py  the eval corpus; DISJOINT from train/corpus/negatives.py
      generate_positives.py  the synthetic speed sweep
      backends.py            one streaming contract over the DEPLOYMENT runtime
      eval_model.py          gates, per-category false accepts, latency
      compare_models.py      matched false accepts
      check_model_alignment.py  openWakeWord window alignment - NOT on backends.py,
                                see the correction in phase 3
    preflight/               4 - live-mic preflight
      test_model.py
    tools/                   not in the pipeline: audit_voices, bench_tts, measure_voice_f0
    docker/                  Dockerfile{,.mww,.piper,.eval}
    scripts/                 setup*.sh, run-training.sh

Everything is run as a module from the repo root - `python -m train.oww.train`,
`python -m eval.compare_models`. **Two path anchors had to move with the files and
neither would have failed loudly:** `train/oww/train.py` chdir'd to its own directory
(now `parents[2]`, the repo root) and `scripts/{run-training,setup}.sh` cd'd to theirs
(now `..`). A wrong anchor there builds a corpus in the wrong place rather than
raising.

Corpora are siblings under `my_custom_model/<wake_word>/{oww,mww}/`, so neither
trainer reaches into the other's. That nesting also fixed a bug: `setup_training_dirs`
rmtree'd `my_custom_model/<wake_word>/`, which is where `scripts/run-training.sh` archives the
commit-tagged models - every run deleted the archive of every previous run.
`patches/configurable-corpus-dir.py` makes openWakeWord read the corpus location from
its config, since upstream derives it from the same value that names the model.

The empty `1_datagen/`, `2_train/`, `3_eval/` directories this file used to complain
about are gone.

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

## Phase 3 - evaluation

**Status: BUILT.** `eval/backends.py`, `Dockerfile.eval`, the `eval` compose service.
`eval_model.py` and `compare_models.py` score both trainers' models through it; the
first mWW model is measured in `tuning_mww.md`.

### The design changed in one important way: don't reimplement inference

The plan above said to generalise `eval/check_model_alignment.py`'s `WakeWordModel` into a
two-backend interface. That was followed, and the first version of `backends.py` also
**reimplemented microWakeWord's inference** - frontend loop, int8 quantization, slice
striding, sliding-window average - from the training repo's source. That was wrong
twice over: it measured a pipeline nothing ships, and every detail it got right had to
be rediscovered by experiment.

**The deployment target is Linux Voice Assistant** (OHF-Voice/linux-voice-assistant),
which implements no inference of its own. It depends on two libraries:

    pymicro-wakeword   MicroWakeWordFeatures + MicroWakeWord
    pyopen-wakeword    OpenWakeWordFeatures  + OpenWakeWord

`backends.py` now drives both exactly as LVA's `__main__.py` does, down to the chunk
size, and computes no features and quantizes no tensors of its own. What is left is
clip handling and the arithmetic that turns probabilities into audio offsets.

Consequences worth stating:

* **The manifest is part of the model.** `MicroWakeWord.from_config()` takes the JSON,
  so scoring the `.json` puts `sliding_window_size` and `probability_cutoff` under test
  with the weights. This is how the dead `probability_cutoff: 1.0` manifest was caught.
* **Both wheels ship `manylinux_2_35_aarch64`**, so the image builds native on Apple
  Silicon and needs no TensorFlow, no ai-edge-litert and no emulation. The base must be
  bookworm or newer: bullseye's glibc 2.31 is below 2.35 and would fall back to an sdist.
* **The openWakeWord side is not on the deployment runtime yet.** `pyopen-wakeword` is
  TFLite-only and the ship candidates are `.onnx`, so `OpenWakeWordOnnxBackend` scores
  them through `openwakeword.model.Model` - comparable with `tuning.md`, not with a
  device. Converting `d1bb9f4` with `onnx2tflite.py` closes this.

### Correction: "point all three tools at it" was wrong, and two is the right number

The plan said to generalise the backend interface and point `compare_models.py`,
`eval_model.py` and `eval/check_model_alignment.py` at it. The first two are on it. **The
third is not, and should not be** - this is a wrong prediction in the plan, not an
outstanding task.

The premise was that all three tools loaded a model the same way, so all three could
share one loader. They do load the same way; that is not what makes a shared interface
possible. What matters is what they FEED it, and the alignment tool feeds something
else entirely:

| | input to the model | question |
|---|---|---|
| `eval_model.py`, `compare_models.py` | 16 kHz PCM, streamed | what would a detector see live? |
| `eval/check_model_alignment.py` | one window of embeddings from `AudioFeatures.embed_clips`, placed at a fixed offset | where in its window does this model want the phrase? |

`backends.py`'s contract is `score(pcm) -> (scores, offsets)`. The alignment tool never
has PCM at the point it calls the model - it has already computed embeddings, and it
deliberately does NOT stream, so that nothing in the streaming feature pipeline can be
blamed for the result (its own docstring says so). Forcing it through the streaming
contract would delete the property it exists to have.

So `eval/check_model_alignment.py` keeps its own `WakeWordModel`, and the duplication
between them is about fifteen lines of interpreter setup - cheaper than a shared
abstraction that has to serve two different input types and two different questions.

**What this does leave open is trap 4 below**, which is a real gap: there is no
alignment measurement for microWakeWord at all. That is a tool that does not exist
yet, not a tool pointed at the wrong loader - and the plan already said to redesign
rather than port it.

### The four traps, as they actually bit

1. **State leaks between clips - CONFIRMED, and worse than expected.** The obvious fix
   does not work: an interpreter with `reset_all_variables()` called and its inputs
   re-zeroed still scores one clip 1.00000 the first time and 0.99608 every time after,
   with or without XNNPACK. Rebuilding the interpreter per clip is the only version
   measured to be reproducible - which is exactly what `MicroWakeWord.reset()` does,
   with the comment "Need to reload model to reset intermediary results". Using the
   deployment runtime made the problem disappear rather than needing solving.

   The self-check (`python -m eval.backends --model M`) compares **full score traces**
   in two corpus orders, not peaks. The leak moved fine scores by 1/255 and left most
   peaks at exactly 1.0, so a peak comparison passed the broken version.

2. **"Threshold" is two parameters - CONFIRMED.** `sliding_window_size` defaults to the
   manifest's value, is overridable, and is printed with every result.

3. **int8 coarsens the scores - CONFIRMED, but the real problem was the FLOOR, not the
   resolution.** Resolution is 0.00078, fine enough. What matters is that the model's
   resting score is 0.245, so **every negative in the corpus fires at 0.15 and below** -
   the threshold openWakeWord ships at. The sweep was extended upwards to 0.95;
   microWakeWord's usable range is 0.25 up, not 0.01 up. See `tuning_mww.md`.

4. **Alignment means something different - NOT ADDRESSED, and it is the one real gap
   left in this phase.** `eval/check_model_alignment.py` remains openWakeWord-only, for the
   reason in the correction above. Latency from end of speech IS measured for both by
   `eval_model.py`, and that is the deployed quantity - but where in its window a
   streaming detector wants the phrase has no tool, so the diagnosis that produced run
   15's alignment fix has no microWakeWord equivalent. A new tool, not a port.

The eval corpora themselves - `my_real_samples_holdout/`, `negatives_tts/` - carried
over untouched, as predicted.

### What is left in phase 3

- **Trap 4**: an alignment measurement for microWakeWord.
- **`d1bb9f4` on the deployment runtime**: convert with `onnx2tflite.py` so the
  headline comparison stops being cross-runtime. Deliberately deferred.

Everything else in this phase is built and exercised.

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

**Actual:** phase 1 → phase 2 → phase 3, with phase 0 skipped.

The argument for the planned order was: **define the eval interface before building
the trainer**, because models are cheap and trustworthy measurement is not, and eight
runs here were judged on contaminated numbers before anyone noticed. Building the
trainer first means the first mWW model arrives with no way to tell whether it is any
good - and the temptation is to judge it at a fixed threshold, which is the mistake
`tuning.md` opens by warning about.

That is exactly what happened. The first mWW model came out with
`average_viable_recall = 0.000` at every step and looked excellent by accuracy,
recall, precision and AUC. Then it was trained a second time, correctly, and arrived
with nothing able to score it - `compare_models.py`, `eval_model.py` and
`check_model_alignment.py` all loaded through `openwakeword.model.Model`, which cannot
read a streaming tflite with internal state.

**What the inverted order actually cost, now that phase 3 is built:** a manifest that
shipped `probability_cutoff: 1.0`, at which the model detects nothing. It was written
by `mww/manifest.py` from a synthetic row in the ROC table, it is a legal value the
runtime loads without complaint, and it survived because there was no way to score the
model it described. That is the same class of failure as `average_viable_recall =
0.000`, from the same cause, one phase later.

Skipping phase 0 cost six avoidable failures (phase 2). Inverting phase 3 cost this
one. **Do not train another mWW configuration without scoring the previous one.**
