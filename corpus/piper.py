"""Piper sample generation over Wyoming TTS.

microWakeWord generates its positives with Piper, so this is the second engine the
shared corpus layer needs. It is deliberately shaped like the Kokoro path in
train.py - render a phrase at a spread of speeds across a spread of voices, write
16 kHz mono WAVs into a directory - so both trainers can consume either engine's
output, or both at once.

THE SPEED PROBLEM. Wyoming's `synthesize` event carries no rate control, so speed
has to be applied after synthesis. audit_voices.py:191 does it by resampling, which
moves pitch as well as rate - correct there, where the point is to deny the ASR a
comfortable rendering, and wrong here. `PLAIN_SPEEDS = (0.7, 1.6)` in train.py means
DELIVERY RATE: Kokoro's speed parameter re-times the phrase without turning the
speaker into a chipmunk, and the model is meant to learn that the phrase can be
said quickly, not that it can be said by someone with a shorter vocal tract. Pitch
is already covered, separately and on purpose, by add_child_range_copies.

So speed here goes through `time_stretch` (WSOLA, pitch-preserving) instead. Using
resampling would silently entangle the speed sweep with the child-range lever and
make run 13's result impossible to attribute.

PIPER IS STOCHASTIC. VITS samples noise and durations per call, so the same
(voice, speaker, text, speed) renders differently every time - measured in
audit_voices.py, where one speaker scored 0% and then 100% across passes. For a
corpus that is free diversity and needs no special handling. For the audit it is
why `--repeats` exists. It also means the corpus is not reproducible from a seed,
which is already true of the Kokoro path (train.py sets no seed).

UNTESTED AGAINST A LIVE SERVICE. Written from the Wyoming protocol as
audit_voices.py implements it, but not yet run against a Piper server - there is
none reachable from where this was written. Phase 0 in plan.md is where it gets
exercised for the first time.
"""

import json
import socket
import uuid
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from scipy.signal import resample_poly
from tqdm import tqdm

from .augment import time_stretch

SR = 16000

# The Wyoming JSONL-over-TCP framing: one JSON header line per event, then
# `data_length` bytes of JSON and `payload_length` bytes of audio.
#
# Duplicated from audit_voices.py rather than shared. That script is standalone by
# design - it runs on the host against live TTS and ASR services and takes no
# dependency on this package - and collapsing the two would drag the corpus layer
# into it. Worth revisiting if a third caller appears.


def _send(sock, etype, data=None, payload=None):
    header = {"type": etype}
    if data is not None:
        header["data"] = data
    if payload is not None:
        header["payload_length"] = len(payload)
    sock.sendall((json.dumps(header) + "\n").encode())
    if payload is not None:
        sock.sendall(payload)


def _read_event(sock, buf):
    """Next event, plus the remaining buffer and this event's audio payload."""
    while b"\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            return None, buf, b""
        buf += chunk
    line, _, buf = buf.partition(b"\n")
    header = json.loads(line)
    n = header.get("data_length") or 0
    while len(buf) < n:
        buf += sock.recv(65536)
    data = json.loads(buf[:n]) if n else header.get("data", {})
    buf = buf[n:]
    p = header.get("payload_length") or 0
    while len(buf) < p:
        buf += sock.recv(65536)
    return {"type": header.get("type"), "data": data}, buf[p:], buf[:p]


def piper_voices(host, port, languages=("en_US", "en_GB"), max_speakers=0):
    """[(voice, speaker_or_None), ...] for the requested languages.

    `max_speakers` caps how many speakers of a multi-speaker model are sampled -
    en_US-libritts_r-medium alone carries 904, and taking all of them would swamp
    the corpus with one model's phonemisation. The sample is evenly spaced rather
    than the first N, because speaker ids are ordered by the source corpus and the
    head of that list is not representative.
    """
    sock = socket.create_connection((host, port), timeout=30)
    try:
        sock.settimeout(30)
        _send(sock, "describe")
        event, _, _ = _read_event(sock, b"")
        info = event["data"] if event else {}
    finally:
        sock.close()

    out = []
    for program in info.get("tts", []):
        for voice in program.get("voices", []):
            langs = voice.get("languages") or [voice.get("language")]
            if languages and not any(str(l).startswith(tuple(languages)) for l in langs):
                continue
            speakers = [s.get("name") for s in (voice.get("speakers") or [])]
            if not speakers:
                out.append((voice["name"], None))
                continue
            if max_speakers and len(speakers) > max_speakers:
                idx = np.linspace(0, len(speakers) - 1, max_speakers).astype(int)
                speakers = [speakers[i] for i in sorted(set(idx))]
            out.extend((voice["name"], s) for s in speakers)
    return out


def piper_render(host, port, voice, speaker, text, speed=1.0):
    """Synthesize one phrase, returned as 16 kHz mono int16.

    `speed` > 1 is faster. Applied with time_stretch after resampling to 16 kHz, so
    it changes delivery rate without moving pitch - see the module docstring.
    """
    sock = socket.create_connection((host, port), timeout=120)
    try:
        sock.settimeout(120)
        v = {"name": voice}
        if speaker is not None:
            v["speaker"] = str(speaker)
        _send(sock, "synthesize", {"text": text, "voice": v})
        buf, pcm, rate = b"", b"", 22050
        while True:
            event, buf, payload = _read_event(sock, buf)
            if event is None:
                break
            if event["type"] in ("audio-start", "audio-chunk"):
                rate = event["data"].get("rate", rate)
                pcm += payload
            elif event["type"] == "audio-stop":
                break
    finally:
        sock.close()

    audio = np.frombuffer(pcm, dtype=np.int16)
    if audio.size == 0:
        return np.zeros(0, dtype=np.int16)

    if rate != SR:
        # resample_poly rather than resample: rational up/down, no FFT-length
        # sensitivity, and it is what vocal_tract_shift already uses.
        from fractions import Fraction
        frac = Fraction(SR, int(rate)).limit_denominator(1000)
        audio = resample_poly(audio.astype(np.float64), frac.numerator, frac.denominator)

    if abs(speed - 1.0) > 1e-3:
        # speed 1.6 = 1.6x faster = 1/1.6 the duration.
        audio = time_stretch(np.asarray(audio, dtype=np.float64), 1.0 / float(speed), sr=SR)

    return np.clip(audio, -32768, 32767).astype(np.int16)


def generate_piper_samples(host, port, voices, output_dir: Path, texts, speeds,
                           samples_per_voice=1, desc="Piper"):
    """Render `texts` across `voices` and `speeds` into `output_dir`.

    Filenames are `piper_{voice}_{speaker}_{uuid}.wav`. Note that this does NOT
    match the `kokoro_`/`runon_` prefixes add_child_range_copies looks for, so Piper
    clips are skipped by the child-range lever rather than mis-shifted - Piper voice
    names carry no sex marker to pick a ratio from. See corpus/augment.py.

    VOICE IS THE OUTER LOOP ON PURPOSE - DO NOT REORDER. wyoming-piper holds exactly
    one loaded voice in a module-level global and reloads it whenever a request names
    a different one (handler.py:333-346, `if voice_name != _VOICE_NAME`). Iterating
    texts or speeds outside voices would rebuild the InferenceSession on every single
    request - and under --use-cuda that means a fresh CUDA session each time, which
    is far more expensive than the synthesis itself.

    The same global is why one server serves strictly one request at a time, and why
    client concurrency measured as pure queueing (docker-compose.yml). Parallelism
    has to come from separate instances, each with its own voice - which also means
    sharding a multi-voice corpus BY VOICE across instances, never round-robin.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(v, s, t, sp)
            for (v, s) in voices
            for t in texts
            for sp in speeds
            for _ in range(samples_per_voice)]

    written = 0
    for voice, speaker, text, speed in tqdm(jobs, desc=desc, unit="clip"):
        try:
            audio = piper_render(host, port, voice, speaker, text, speed)
        except Exception as e:
            print(f"  Error rendering {voice}/{speaker} at {speed}x: {e}")
            continue
        if audio.size < 480:
            continue
        tag = f"{voice}_{speaker}" if speaker is not None else voice
        name = f"piper_{tag}_{uuid.uuid4().hex[:8]}.wav".replace("/", "_")
        scipy.io.wavfile.write(str(output_dir / name), SR, audio)
        written += 1

    print(f"  Wrote {written} Piper clips from {len(jobs)} jobs")
    return written
