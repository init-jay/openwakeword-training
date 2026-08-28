# OpenWakeWord Trainer

Train custom wake word models for [OpenWakeWord](https://github.com/dscripka/openWakeWord) using synthetic voices from Kokoro TTS combined with your real voice recordings.

**Why this exists:** The official OpenWakeWord training process relies on Google Colab notebooks that frequently break. This repo provides a working local training pipeline that produces quality models.

## What You Get

- A trained `.onnx` wake word model (~400KB), convertible to `.tflite` (see `onnx2tflite.py`)
- Works with OpenWakeWord, Home Assistant, wyoming-openwakeword, or anything that loads ONNX or tflite

### Typical results

Measured for "hey seeree" against 56 held-out real recordings and a 100-clip synthetic negative corpus, at threshold 0.5 (`eval_model.py`):

| | result |
|---|---|
| detection, phrase spoken alone | 53/56 (95%) |
| detection, command spoken immediately after | 53/56 (95%) |
| median latency from end of speech | 77 ms |
| false accepts, general conversation | 0/36 |
| false accepts, bare commands (no wake word) | 0/12 |
| false accepts, other assistants ("hey Google", "Alexa") | 0/8 |
| false accepts, "hey" + a different name | 0/12 |
| **false accepts, phrase continuing into another word** | **6/20** |

The last row is the honest caveat: this model is quiet on ordinary speech but still fires on close phonetic neighbours ("hey serious", "hey series"). Those clips are adversarial by construction — a fifth of the negative corpus — so read the categories separately rather than pooling them into one false-accept rate.

Results depend heavily on the wake word and on how many real recordings you provide (91 clips from 2 speakers here). `tuning.md` documents the measurements behind these numbers and what moved them.

## Requirements

- **NVIDIA GPU** with CUDA (RTX 3060 12GB or better recommended)
- **Docker** with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- **~20GB disk space** for training data

## Quick Start (Docker)

Docker is the recommended approach - it handles all the dependency hell for you.

### 1. Clone

```bash
git clone https://github.com/CoreWorxLab/openwakeword-training.git
cd openwakeword-training
```

### 2. Download Training Data (~17GB, one-time)

```bash
docker compose build trainer
docker compose run --rm trainer ./setup-data.sh
```

### 3. Record Your Voice (Optional but Recommended)

Recording 20-50 samples of your actual voice significantly improves detection. This runs on your host machine (needs microphone access):

```bash
pip install pyaudio numpy scipy
python record_samples.py --wake-word "hey cal"
```

- Press ENTER, then wait for "SPEAK NOW!" before speaking
- Say your wake word naturally
- Vary your tone, speed, and distance from the mic
- Press 'q' to quit

The cue fires only after the microphone has opened and settled (0.6s warm-up). Speaking before it means the word onset is never captured.

**Multiple speakers:** put each person in their own subdirectory — `train.py` searches `my_real_samples/` recursively and flattens the path into the training filename, so speakers with identical clip names don't collide:

```
my_real_samples/
├── jay/    hey_seeree_0001.wav …
└── alex/   hey_seeree_0001.wav …
```

Loose files directly in `my_real_samples/` still work. Keeping speakers separate lets you re-record or drop one without guessing which clips were whose.

### 3b. Check Your Recordings (Optional)

Before committing to a multi-hour training run, verify your samples are usable:

```bash
python check_alignment.py my_real_samples/ --verbose
```

This reports how far each clip's speech sits from the end of OpenWakeWord's detection window, and flags clips with no detectable speech. Training trims silence automatically, so large trailing-silence numbers here are expected and fine — what matters is that speech is actually present and not clipped.

### 4. Train Your Model

```bash
docker compose run --rm trainer python train.py --wake-word "hey cal" --data-dir /app/data
```

### Training speed

Measured on an RTX 3090 with 42 Kokoro voices at `--samples-per-voice 300` (~20K positive and ~13K negative clips, 3 augmentation rounds, 50K steps):

| stage | time | share |
|---|---:|---:|
| TTS generation (2 Kokoro containers) | ~17 min | 49% |
| Feature computation (GPU) | ~4 min | 11% |
| Model training | ~14 min | 40% |
| **total** | **~35 min** | |

How it got there, since the defaults matter:

| | total |
|---|---:|
| CPU feature computation, 1 Kokoro | 83 min |
| + `onnxruntime-gpu` (features 36 → 4 min) | 52 min |
| + a second Kokoro container (TTS 34 → 17 min) | 35 min |

**Why more than one Kokoro container:** a Kokoro process is single-threaded and saturates exactly one core, so concurrent requests to one instance simply queue (4 client threads measured 15.0 it/s against 14.1 sequential) while adding instances scales almost linearly (2 instances: 28.7 it/s) — the GPU sits at ~21% throughout, so cores, not the GPU, are the limit.

`docker-compose.yml` ships two. Add more by copying the `kokoro2` block and appending to `KOKORO_URL`, keeping at least one core free for the trainer; check `nproc` first. Past ~3 instances TTS stops being the slowest stage and the whole-run saving flattens out.

### 5. Test Your Model

Test on your host machine (needs microphone access):

```bash
pip install openwakeword numpy
python test_model.py --list-devices                             # find your mic
python test_model.py --model my_custom_model/hey_cal.onnx --device 0
```

Speak your wake word into the microphone and watch for detections. Capture goes through ffmpeg, so no PortAudio/PyAudio install is needed — just `ffmpeg` on your PATH.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--wake-word` | "hey cal" | The wake word/phrase to detect |
| `--samples-per-voice` | 300 | Samples generated per Kokoro voice |
| `--training-steps` | 50000 | More steps = better but slower (~14 min per 50K) |
| `--layer-size` | 64 | Network size (32, 64, or 128) |
| `--kokoro-url` | http://localhost:8880 | Kokoro TTS endpoint. Comma-separate several to split the work across them |
| `--tts-workers` | 2 | Concurrent requests **per server**; total is this times the server count |
| `--augmentation-rounds` | 3 | Differently-augmented copies of each clip. Multiplies training data at no TTS cost |
| `--runon-fraction` | 0.4 | Share of positives where the phrase runs straight into a command rather than silence |
| `--data-dir` | `.` | Training data directory (`/app/data` for Docker) |
| `--no-trim` | off | Skip silence trimming before augmentation (not recommended) |
| `--negatives-file` | — | Confusable negative phrases for a wake word with no built-in list |

## How It Works

1. **Sample Generation** - Creates ~20K positive samples using ~42 English Kokoro voices with speed variation (0.7-1.3x), plus your real recordings (weighted 3x). A configurable share (40%) are "run-on" positives where the phrase flows straight into a command, cut just after the wake word using Kokoro's word timestamps

2. **Negative Samples** - Generates both clearly-different phrases ("hello", "alexa") and near-misses of the wake word, plus the commands used in the run-on positives on their own

3. **Silence Trimming** - Strips leading/trailing silence from all samples so speech lands where the model expects it (see below)

4. **Augmentation** - OpenWakeWord adds noise, reverb, and mixing to simulate real-world conditions

5. **Training** - Neural network learns to distinguish your wake word from everything else

### Key Insight

**Do use similar-sounding negatives** — this reverses earlier advice in this README, which measurement contradicted. A model trained only on clearly-different phrases rejects exactly what it was shown and nothing adjacent: it scored 0/8 on other assistants' wake words and 0/36 on general conversation, but false-accepted 13/20 on the phrase continuing into another word ("hey serious" → 0.995) and 5/12 on "hey" plus a different name. Adding near-misses to the wordlist cut that to 4/20 on held-out confusables.

`train.py` keeps these in `CONFUSABLE_NEGATIVES`, keyed by wake word; use `--negatives-file` for a phrase with no built-in list. It warns if you train without any. See `tuning.md` for the full measurements.

### Why Silence Trimming Matters

OpenWakeWord's `create_fixed_size_clip` aligns the **end of the array** with the end of the detection window, not the end of the speech. Untrimmed, a tight ~1.3s Kokoro clip and a fixed 2s recording place their speech at completely different offsets — so the two halves of the positive set teach contradictory alignments, and the real recordings are weighted 3x. Trimming first collapses that difference. Both positives and negatives are trimmed, so clip length never becomes a cue the model can learn instead of the phrase.

## Output

```
my_custom_model/
├── hey_cal.onnx          # Your trained model - use this!
└── hey_cal/
    ├── positive_train/   # Generated training samples
    ├── positive_test/    # Test samples
    ├── negative_train/   # Negative training samples
    └── negative_test/    # Negative test samples
```

## Using Your Model

```python
from openwakeword.model import Model

model = Model(wakeword_models=["my_custom_model/hey_cal.onnx"])

# Process 16kHz mono audio frames
prediction = model.predict(audio_frame)
if prediction["hey_cal"] > 0.5:
    print("Wake word detected!")
```

## Manual Setup (No Docker)

If you prefer not to use Docker, you can set up the environment directly:

```bash
./setup.sh
source venv/bin/activate

# Start Kokoro TTS separately
docker run -d --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest

python train.py --wake-word "hey cal"
```

Note: This requires Python 3.10+ and working CUDA. The pinned dependency versions in `requirements.txt` can conflict with other Python packages on your system, which is why Docker is recommended.

## Troubleshooting

### "Reached EOF prematurely" warnings
Normal - Kokoro's WAV headers have a quirk but the audio data is fine.

### Low recall in training metrics
Training metrics use synthetic test samples. Real-world performance is usually better.

### Model not detecting wake word
- Ensure audio is 16kHz mono
- Model needs ~2 seconds of audio buffer to warm up
- Try lowering detection threshold (default 0.5)

### TFLite conversion error at end
Ignore - the ONNX model is saved successfully before this error.

Run this on the jupyter lab box with tf-lab container
```bash
docker compose run --rm \
    -v ~/openwakeword-training:/oww -w /oww \
    jupyter bash -c \
    "pip install -q onnxruntime onnx2tf onnx onnx-graphsurgeon sng4onnx tf_keras psutil ai-edge-litert && \
     python onnx2tflite.py my_custom_model/hey_seeree.onnx \
        -o my_custom_model/hey_seeree.tflite"
```

## Credits

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) by David Scripka
- [Kokoro TTS](https://github.com/remsky/Kokoro-FastAPI) for synthetic voice generation
- Training data from [ACAV100M](https://huggingface.co/datasets/davidscripka/openwakeword_features)

## License

MIT
