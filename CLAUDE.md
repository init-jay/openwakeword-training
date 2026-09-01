# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local training pipeline for custom OpenWakeWord wake word models. Uses Kokoro TTS for synthetic voice generation combined with real voice recordings to produce `.onnx` models.

## Key Commands

### Docker (recommended)
```bash
# REBUILD AFTER PULLING: train.py now imports the corpus/ package, so an image
# built before it existed fails with ModuleNotFoundError: No module named 'corpus'.
docker compose build trainer                  # Build training image
docker compose run --rm trainer ./setup-data.sh  # Download ~17GB training data
docker compose run --rm trainer python train.py --wake-word "hey cal" --data-dir /app/data

# Where does the trained model want the phrase in the window? (needs openwakeword)
docker compose run --rm -v $(pwd)/check_model_alignment.py:/app/check_model_alignment.py \
    trainer python check_model_alignment.py --model /app/my_custom_model/hey_cal.onnx

# Convert to tflite. --no-deps because the conversion is CPU-only and does not
# need the Kokoro GPU services that `trainer` otherwise starts.
docker compose run --rm --no-deps trainer \
    python onnx2tflite.py /app/my_custom_model/hey_cal.onnx -o /app/my_custom_model/hey_cal.tflite
```

### Host (mic access needed)
```bash
cd record_real_sample && uv run record_samples.py --wake-word "hey cal"   # Record voice samples
python check_alignment.py my_real_samples/ --verbose            # Inspect sample timing
python test_model.py --list-devices                             # List microphones
python test_model.py --model my_custom_model/hey_cal.onnx       # Test model
```

### Manual (no Docker)
```bash
./setup.sh                                    # One-time setup (downloads ~17GB)
source venv/bin/activate
python train.py --wake-word "hey cal"
```

## Architecture

**Docker container** handles training (the dependency-heavy part):
- `train.py` orchestrates the full pipeline
- Generates positive/negative WAV samples via Kokoro TTS API (~42 English voices of 67 total, speed 0.7-1.3x). Note ~11 of the English voices are `v0` variants of others, so distinct speakers number closer to 30.
- Copies real voice recordings from mounted `my_real_samples/` (3x weighted)
- Creates `training_config.yaml` from OpenWakeWord's template
- Shells out to `openwakeword/openwakeword/train.py --augment_clips` then `--train_model`
- Outputs `.onnx` model to mounted `my_custom_model/`

**Host** handles mic-dependent tasks:
- `record_real_sample/record_samples.py` - records real voice samples. Own uv env; captures via ffmpeg (no PyAudio), reports peak/noise/SNR per take, and numbers files from the highest existing index so gaps never overwrite
- `test_model.py` - live mic testing of trained models (streams raw PCM from ffmpeg's avfoundation input; no PyAudio)
- `check_alignment.py` - reports where speech sits in the detection window (numpy + scipy only, so it runs anywhere)
- `check_model_alignment.py` - the same question asked of a *trained* model: sweeps where the phrase is placed and reports the alignment the model learned, which is also its latency floor. Needs onnxruntime and an importable `openwakeword`, so run it in the trainer container (or with `PYTHONPATH` pointing at an openWakeWord checkout).

### Docker volume mounts
- `./data` → `/app/data` - 17GB feature files, background audio, impulse responses
- `./my_real_samples` → `/app/my_real_samples` - user voice recordings
- `./my_custom_model` → `/app/my_custom_model` - trained model output

### Kokoro TTS
- Runs as separate Docker service via `docker-compose.yml`
- trainer container connects via `http://kokoro:8880` (set by KOKORO_URL env var)
- train.py also accepts `--kokoro-url` flag

### Piper TTS
- `Dockerfile.piper` + the `piper` service. Built rather than pulled:
  `rhasspy/wyoming-piper` is debian-slim with CPU onnxruntime and cannot use a GPU.
- Wyoming protocol over TCP (10200), not HTTP. Client is `corpus/piper.py`.
- Start it explicitly - the trainer deliberately does not `depends_on` it:
  `docker compose up -d piper`
- **No `--use-cuda`: the GPU makes it 2.5x SLOWER** (17.45 clips/s CPU vs 7.00 with
  CUDA, measured back to back). CUDA does initialise; ORT then splits the VITS graph,
  leaving shape ops on CPU and paying 28 Memcpy nodes per inference. See
  `docker-compose.yml` for the numbers and the ORT warnings.
- **ONE instance, unlike Kokoro's two** - for two independent reasons. Within an
  instance wyoming-piper holds one voice in a global, so requests serialise (latency
  is exactly proportional to queue depth). Across instances they oversubscribe ORT's
  thread pool: a second one measured **0.60x** on CPU, a 40% loss.
- ~17.9 clips/s, slightly faster per clip than one Kokoro instance: 10k clips in
  about 10 minutes. Not worth optimising further.
- `bench_tts.py` re-measures this for either engine. It needs numpy/scipy/tqdm, so
  the reliable way is inside the trainer container (the host has python3 but not
  necessarily the deps), reaching the service by its compose name rather than
  localhost:

```bash
docker compose --profile bench up -d piper piper2   # piper2 only for --instances
docker compose ps                                   # both should say "running"
docker compose run --rm --no-deps \
    -v $(pwd)/bench_tts.py:/app/bench_tts.py -v $(pwd)/corpus:/app/corpus \
    trainer python3 bench_tts.py --engine piper --host piper --port 10200 \
    --instances piper:10200 piper2:10200
docker compose --profile bench stop piper2          # stop, NOT down - `down` is
                                                    # project-scoped and takes the
                                                    # network (and piper) with it
```

Inside the compose network the instances are `piper:10200` and `piper2:10200` - the
`10201` mapping only exists on the host. From the host it would be
`--instances localhost:10200 localhost:10201` instead, if numpy/scipy/tqdm are
available there.

```
```
- `--use-cuda` is in the compose `command`, so it can be A/B'd without a rebuild.
  Unmeasured, but the case is decent: Piper is genuinely CPU-bound, so offload has
  something to win, unlike Kokoro where the GPU already sat idle at 21%.

## Important Design Decisions

- **Negatives must include near-misses of the wake word**, not just clearly different phrases. Measured on `hey_seeree.onnx` (see `tuning.md`): trained on nine distinct phrases only, it false-accepted 0/8 on other assistants and 0/36 on general conversation but 13/20 on the phrase continuing into another word ("hey serious" → 0.995) and 5/12 on "hey" plus another name. `train.py` therefore keeps `BASE_NEGATIVES` (wake-word independent) plus `CONFUSABLE_NEGATIVES[safe_name]`; `--negatives-file` supplies the latter for a wake word with no built-in entry.
- **Training negatives are kept disjoint from the eval corpus** in `generate_negatives.py`. The false-accept gates in `tuning.md` are scored on that corpus, so a phrase in both would turn a generalisation measurement into a memorisation one.
- **Some Kokoro voices mispronounce the wake word, and must be excluded by ear.** A wake word is not a dictionary word, so g2p guesses at it and voices guess differently — 6 of 42 said something other than "hey seeree", ~14% of the synthetic corpus. `MISPRONOUNCING_VOICES` holds them per wake word; `--exclude-voices` adds more. **Rebuild this list for every new wake word by rendering all 42 voices saying the phrase and listening to each.** Duration is not a usable proxy: the worst offender sat at exactly the median length. Same rule applies to `positive_texts` — never add a variant without hearing it.
- **Positives are pitch/formant-shifted to cover non-adult voices.** `add_child_range_copies` adds a resampled-and-restretched copy of a fraction of the Kokoro clips (`--child-fraction`, default 0.5), at a ratio picked from the voice's sex: female 1.20-1.35x, male 1.15-1.30x. Without it the corpus is adult-only and a child is barely detected at all — run 12 measured a 4-year-old at 24% against the adult's 97%, and at 34% on his *own training clips*. Copies are added, never substituted, so adult density is unchanged.
- **Evaluate every speaker separately, never pooled.** `my_real_samples_holdout/<speaker>/` and `<speaker>_runon/`. A speaker missing from the holdout does not produce a worse number, it produces a confident irrelevant one — that is how the child gap stayed invisible for eleven runs.
- `max_negative_weight` (`train.py`, currently 2000) is the lever if a larger negative list makes the model too conservative.
- All audio is 16kHz, 16-bit, mono WAV.
- Real voice samples are copied 3x to weight them higher in training.
- **`my_real_samples/` is searched recursively**, so speakers can live in per-speaker subdirectories (`my_real_samples/jay/`). The relative path is flattened into the destination filename — two speakers recording the same phrase produce identical basenames, so using the basename alone would silently overwrite one speaker's clips with the other's.
- **Silence is trimmed from all samples before augmentation.** OpenWakeWord's `create_fixed_size_clip` (`openwakeword/data.py:719`) aligns the end of the *array* — not the end of the *speech* — with the end of the detection window, so untrimmed silence displaces the phrase. Untrimmed, tight Kokoro clips and fixed 2s recordings land at different offsets and teach contradictory alignments. Negatives are trimmed too, so clip length can't become a class cue. Disable with `--no-trim`.
- **`record_real_sample/record_samples.py` cues after a 0.6s mic warm-up.** Cueing before the input device settles loses the first fraction of a second and clips the word onset.
- `--data-dir` flag lets train.py work both inside Docker (`/app/data`) and on host (`.`).
