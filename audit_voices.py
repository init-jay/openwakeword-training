#!/usr/bin/env python3
"""
Find TTS voices that say the wrong thing, before they poison the training corpus.

WHY THIS EXISTS
---------------
A wake word worth having is not a dictionary word, so a TTS engine's grapheme-to-
phoneme has to guess at it - and voices guess differently. Six of Kokoro's 42 English
voices do not say "hey seeree" (af_alloy, am_echo, bf_alice, bf_lily, bm_daniel,
bm_fable). Every clip such a voice produces is a mislabelled positive, and at 1/42 of
the voice list each that was ~14% of the synthetic corpus for the six together.

They were found by listening to all 42, which is the only method that had worked.
Duration is not a proxy: bm_fable sits at exactly the median length and is wrong,
while am_onyx is 22% below median and is fine. That does not scale to a Piper voice
list with hundreds of speakers, which is what this script is for.

HOW IT WORKS
------------
Render the phrase at several speeds per voice, transcribe each with ASR, and compare
the FINAL TOKEN of each transcript against the consensus across the whole voice set.

The final token is the discriminator, not string similarity. ASR normalises an
invented word toward a real one - every good voice here transcribes as "hey siri" -
so whole-string similarity puts bf_lily's "hey sorry" at 0.82 against "hey siri" and
cannot separate it from af_sarah's "ahay siri" at 0.88. Comparing only the last token
(sorry != siri, siri == siri) separates them cleanly, and ignores the leading filler
ASR invents on slow renderings.

SEVERAL SPEEDS ARE REQUIRED. At 1.0x alone, bf_lily transcribes as "Hey, siri." and
bm_fable as "Hey Siri." - both pass. At 0.75x they are "Hey, sorry." and
"Hey, Sairee." A marginal pronunciation only separates from the real word when the
ASR's language model has less room to smooth it over.

Validated against the six known-bad voices: all six flagged, and the known-good ones
matched consensus at every speed.

THIS IS A SCREEN, NOT A VERDICT. It produces a ranked shortlist to check by ear. A
voice at 100% is probably fine; anything below it needs listening to before it is
trusted or excluded.

Usage:
    python audit_voices.py --wake-word "hey seeree" \
        --kokoro-url http://192.168.2.26:8880 --asr 192.168.2.14:10300

    # after: paste the printed block into MISPRONOUNCING_VOICES in train.py
"""

import argparse
import json
import re
import socket
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import requests
import scipy.io.wavfile
from scipy.signal import resample

SR = 16000
DEFAULT_SPEEDS = (0.75, 1.0, 1.3)


# --------------------------------------------------------------------------
# Wyoming ASR client
#
# Wyoming is a JSONL-over-TCP protocol: one JSON header line per event, then
# `data_length` bytes of JSON and `payload_length` bytes of audio. Implemented
# here rather than taking the `wyoming` dependency, which would have to be added
# to the trainer image for a script that never runs inside it.
# --------------------------------------------------------------------------

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
    while b"\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            return None, buf
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
    return {"type": header.get("type"), "data": data}, buf[p:]


def transcribe(pcm16, host, port, timeout=120):
    """Transcribe int16 mono PCM via a Wyoming ASR service."""
    sock = socket.create_connection((host, port), timeout=timeout)
    try:
        sock.settimeout(timeout)
        fmt = {"rate": SR, "width": 2, "channels": 1}
        _send(sock, "transcribe", {"language": "en"})
        _send(sock, "audio-start", {**fmt, "timestamp": 0})
        raw = pcm16.tobytes()
        for i in range(0, len(raw), 2 * SR):
            _send(sock, "audio-chunk", {**fmt, "timestamp": i // 2},
                  payload=raw[i:i + 2 * SR])
        _send(sock, "audio-stop", {"timestamp": len(raw) // 2})
        buf = b""
        while True:
            event, buf = _read_event(sock, buf)
            if event is None:
                return ""
            if event["type"] == "transcript":
                return (event["data"].get("text") or "").strip()
    finally:
        sock.close()


# --------------------------------------------------------------------------
# Kokoro rendering
# --------------------------------------------------------------------------

def kokoro_voices(url):
    raw = requests.get(f"{url}/v1/audio/voices", timeout=30).json().get("voices", [])
    voices = [v["id"] if isinstance(v, dict) else v for v in raw]
    return sorted(v for v in voices if v.startswith(("af_", "am_", "bf_", "bm_")))


def kokoro_render(url, voice, text, speed):
    r = requests.post(f"{url}/v1/audio/speech",
                      json={"model": "kokoro", "voice": voice, "input": text,
                            "speed": speed, "response_format": "wav"}, timeout=120)
    r.raise_for_status()
    import io
    sr, data = scipy.io.wavfile.read(io.BytesIO(r.content))
    if data.ndim > 1:
        data = data[:, 0]
    if sr != SR:
        data = resample(data.astype(np.float64), int(len(data) * SR / sr))
        data = np.clip(data, -32768, 32767)
    return data.astype(np.int16)


# --------------------------------------------------------------------------

def final_token(text):
    """Last alphabetic token, lowercased.

    ASR invents leading filler on slow renderings ("Ahay Siri", "Ah hey Siri") but
    the word being tested is always last, so this ignores the noise and keeps the
    signal.
    """
    words = re.findall(r"[a-z']+", text.lower())
    return words[-1] if words else ""


def main():
    p = argparse.ArgumentParser(
        description="Screen TTS voices for mispronunciation of the wake word")
    p.add_argument("--wake-word", required=True)
    p.add_argument("--kokoro-url", default="http://localhost:8880")
    p.add_argument("--asr", default="localhost:10300",
                   help="Wyoming ASR service, host:port (default: %(default)s)")
    p.add_argument("--speeds", default=",".join(str(s) for s in DEFAULT_SPEEDS),
                   help="Comma-separated render speeds. More speeds catch more "
                        "marginal voices; 1.0 alone misses them (default: %(default)s)")
    p.add_argument("--voices", default="",
                   help="Comma-separated subset to audit (default: all English)")
    p.add_argument("--out-dir", default="voice_audit",
                   help="Where to write clips for the ear check (default: %(default)s)")
    args = p.parse_args()

    host, _, port = args.asr.partition(":")
    port = int(port or 10300)
    speeds = [float(s) for s in args.speeds.split(",") if s.strip()]

    voices = ([v.strip() for v in args.voices.split(",") if v.strip()]
              or kokoro_voices(args.kokoro_url))
    if not voices:
        print("No voices found.")
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Auditing {len(voices)} voices x {len(speeds)} speeds "
          f"= {len(voices) * len(speeds)} renderings")
    print(f"  TTS {args.kokoro_url}   ASR {host}:{port}\n")

    results = {}
    for i, voice in enumerate(voices, 1):
        rows = []
        for speed in speeds:
            try:
                audio = kokoro_render(args.kokoro_url, voice, args.wake_word, speed)
            except Exception as e:
                print(f"  {voice} @ {speed}: render failed ({e})")
                continue
            text = transcribe(audio, host, port)
            rows.append((speed, text, len(audio) / (SR / 1000)))
            scipy.io.wavfile.write(str(out / f"{voice}_{speed}.wav"), SR, audio)
        results[voice] = rows
        print(f"\r  {i}/{len(voices)} {voice:16s}", end="", flush=True)
    print("\r" + " " * 40 + "\r", end="")

    # Consensus is the most common final token across every rendering. It is what
    # the ASR reliably hears for a correct pronunciation - "siri" for "hey seeree" -
    # not the wake word itself, which it will never spell right.
    tokens = [final_token(t) for rows in results.values() for _, t, _ in rows]
    tokens = [t for t in tokens if t]
    if not tokens:
        print("ASR returned nothing for any voice - is the service reachable?")
        return 1
    consensus, n_consensus = Counter(tokens).most_common(1)[0]
    print(f'Consensus final token: "{consensus}" '
          f"({n_consensus}/{len(tokens)} renderings)\n")

    durations = [d for rows in results.values() for _, _, d in rows]
    median_ms = float(np.median(durations))

    scored = []
    for voice, rows in results.items():
        if not rows:
            continue
        hits = sum(1 for _, t, _ in rows if final_token(t) == consensus)
        scored.append((hits / len(rows), voice, rows))
    scored.sort(key=lambda r: (r[0], r[1]))

    print(f"{'voice':16s} {'agree':>6s} {'dur':>7s}  transcripts")
    print("-" * 96)
    suspect = []
    for frac, voice, rows in scored:
        dur = np.median([d for _, _, d in rows])
        texts = " | ".join(t or "(silence)" for _, t, _ in rows)
        mark = "" if frac == 1.0 else ("  <-- BAD" if frac < 0.5 else "  <-- CHECK")
        if frac < 1.0:
            suspect.append(voice)
        print(f"{voice:16s} {frac*100:5.0f}% {dur:6.0f}ms  {texts[:60]:60s}{mark}")

    print(f"\nmedian duration {median_ms:.0f}ms. Clips written to {out}/")
    if suspect:
        print(f"\n{len(suspect)} voice(s) to check BY EAR - this is a screen, not a "
              f"verdict.\nListen to {out}/<voice>_*.wav, then paste the ones that are "
              f"genuinely wrong:\n")
        safe = args.wake_word.replace(" ", "_").lower()
        print(f'    "{safe}": [')
        print("        " + ", ".join(f'"{v}"' for v in sorted(suspect)) + ",")
        print("    ],")
    else:
        print("\nEvery voice matched consensus at every speed. Still worth spot-"
              "checking a few by ear before trusting a new engine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
