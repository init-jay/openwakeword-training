# Tuning guidance for wake-word training

Derived from measuring `hey_seeree.onnx` against 56 real recordings and a 100-clip
synthetic negative corpus. Every number below is measured, not estimated; the method
is at the end so it can be re-run after a retrain.

Priorities are ordered by measured impact per unit of effort, not by how interesting
they are.

---

## What the current model does

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

**Diagnose before changing anything:**

```bash
python check_alignment.py my_custom_model/hey_seeree/positive_train --verbose
```

If it reports ~440 ms, the trimming is not doing what it should for this corpus — check
whether the model predates trimming, whether `--no-trim` was used, or whether Kokoro's
output has a tail that `top_db=40` does not catch. If it reports ~130 ms, then the
model's preference was learned from something else and the theory needs revisiting.

**Target:** phrase ending 100-150 ms before the window end. Worth ~100-150 ms of
latency. Do not drive it to zero — see the note under Priority 3.

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
2. Run `check_alignment.py` on the positives and reconcile the 440 ms (Priority 2).
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
python ../openWakeWord/scripts/eval_model.py \
    --model my_custom_model/hey_seeree.onnx \
    --positives my_real_samples/jay \
    --negatives negatives_tts
```

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
