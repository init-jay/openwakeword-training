#!/usr/bin/env python3
"""
Record real voice samples for wake word training.
Creates 16kHz mono WAV files in my_real_samples/.

Usage:
    python record_samples.py --wake-word "hey cal"
"""

import argparse
import os
import time
import wave
from ctypes import CFUNCTYPE, POINTER, c_char_p, c_int, cdll
from pathlib import Path

import numpy as np

# Suppress ALSA warnings on Linux
try:
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def _alsa_error_handler(filename, line, function, err, fmt):
        pass
    _c_error_handler = ERROR_HANDLER_FUNC(_alsa_error_handler)
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(_c_error_handler)
except Exception:
    pass

import pyaudio

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
DURATION = 2.0  # seconds of usable audio captured after the cue
WARMUP = 0.6    # seconds discarded after opening the device, before the cue

FULL_SCALE = 32768.0
# Speech should peak somewhere near -12 dBFS. Much below that and the recording is
# wasting dynamic range: the first corpus recorded with this script came in at a
# median peak of -22 dBFS with a -26 dB noise floor, which is audibly crackly and
# cannot be repaired afterwards - amplifying a quiet clip raises its noise with it,
# so SNR is fixed at capture. The only fix is more gain while recording.
LOW_LEVEL_DBFS = -18.0
CLIPPING_DBFS = -0.5
LOW_SNR_DB = 20.0


def measure(speech: np.ndarray, noise: np.ndarray):
    """Peak level, noise floor and SNR, all in dB.

    Speech level is the 90th-percentile 10 ms frame rather than the peak, so one
    transient does not stand in for the whole utterance. The noise floor comes from
    the warm-up audio, which is recorded before the cue and is therefore room tone
    by construction - it costs nothing to measure and is the only honest reference
    for how much of the recording is not signal.
    """
    peak = int(np.abs(speech).max()) if speech.size else 0
    peak_dbfs = 20 * np.log10(max(peak, 1) / FULL_SCALE)

    frame = SAMPLE_RATE // 100
    n = len(speech) // frame
    if n < 2:
        return peak_dbfs, None, None, 0
    frames = speech[:n * frame].astype(np.float64).reshape(n, frame)
    speech_rms = float(np.percentile(np.sqrt(np.mean(frames ** 2, axis=1)), 90))

    noise_rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2))) if noise.size else 0.0
    snr = 20 * np.log10(speech_rms / noise_rms) if noise_rms > 0 and speech_rms > 0 else None
    noise_dbfs = 20 * np.log10(noise_rms / FULL_SCALE) if noise_rms > 0 else None

    clipped = int((np.abs(speech) >= 32700).sum())
    return peak_dbfs, noise_dbfs, snr, clipped


def record_sample(filename: str, p: pyaudio.PyAudio):
    """Record one sample, cueing once the stream is live. Returns its levels."""
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    # An input device does not deliver usable audio the instant it opens. Read
    # WARMUP seconds first so the cue below lands on an already-running stream -
    # cueing before the device settles loses the first fraction of a second of
    # speech and clips the word onset (the "h" in "hey"). This audio is not written
    # to the sample, but its second half is kept as a noise-floor reference; the
    # first half can still hold the device settling.
    warmup = []
    for _ in range(int(SAMPLE_RATE / CHUNK * WARMUP)):
        warmup.append(stream.read(CHUNK, exception_on_overflow=False))

    print("SPEAK NOW!")

    frames = []
    for _ in range(int(SAMPLE_RATE / CHUNK * DURATION)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))

    speech = np.frombuffer(b''.join(frames), dtype=np.int16)
    noise = np.frombuffer(b''.join(warmup[len(warmup) // 2:]), dtype=np.int16)
    return measure(speech, noise)


def level_report(peak_dbfs, noise_dbfs, snr, clipped):
    """One line of feedback, plus any warning worth acting on before the next take."""
    parts = [f"peak {peak_dbfs:>6.1f} dBFS"]
    if noise_dbfs is not None:
        parts.append(f"noise {noise_dbfs:>6.1f} dBFS")
    if snr is not None:
        parts.append(f"SNR {snr:>4.0f} dB")
    print("    " + "   ".join(parts))

    if clipped:
        print(f"    CLIPPING: {clipped} sample(s) at full scale - lower the input gain.")
    elif peak_dbfs > CLIPPING_DBFS:
        print("    Very close to full scale - lower the input gain.")
    elif peak_dbfs < LOW_LEVEL_DBFS:
        print(f"    LOW LEVEL: aim for about -12 dBFS. Raise the input gain or move "
              f"closer;\n    this cannot be fixed later, since amplifying the clip "
              f"raises its noise too.")
    if snr is not None and snr < LOW_SNR_DB:
        print(f"    NOISY: only {snr:.0f} dB above the room. Quieter room, or closer mic.")


def main():
    parser = argparse.ArgumentParser(description="Record voice samples for wake word training")
    parser.add_argument("--wake-word", default="hey cal", help="Wake word you're recording")
    parser.add_argument("--output-dir", default="my_real_samples", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    safe_name = args.wake_word.replace(" ", "_").lower()

    print("=" * 50)
    print(f"Voice Sample Recorder - \"{args.wake_word}\"")
    print("=" * 50)
    print()
    print("Record at least 20-50 samples for best results.")
    print("Vary your tone, speed, distance from mic, etc.")
    print()
    print("  - Press ENTER to start recording")
    print(f"  - Wait for \"SPEAK NOW!\", then say \"{args.wake_word}\" naturally")
    print("  - Recording lasts 2 seconds; silence is trimmed at training time")
    print("  - Levels are reported after each take; aim for a peak near -12 dBFS")
    print("  - Press 'q' + ENTER to quit")
    print()

    count = len(list(output_dir.glob("*.wav")))
    print(f"Existing samples: {count}")
    print()

    # Created once and reused - instantiating PyAudio enumerates devices, which is
    # slow enough to matter if it happens between the cue and the first read.
    p = pyaudio.PyAudio()
    session = []
    try:
        while True:
            user_input = input(f"[Sample {count + 1}] Press ENTER to record (q to quit): ")

            if user_input.lower() == 'q':
                break

            print("Get ready...", end=" ", flush=True)
            time.sleep(0.5)

            filename = str(output_dir / f"{safe_name}_{count + 1:04d}.wav")
            levels = record_sample(filename, p)

            print(f"Saved: {filename}")
            level_report(*levels)
            session.append(levels)
            count += 1
            print()
    finally:
        p.terminate()

    print(f"\nDone! {count} total samples in {output_dir}/")

    # A per-sample warning is easy to shrug off; the same warning across a whole
    # session is a setup problem, and it is much cheaper to fix now than after
    # discovering it in a trained model.
    if session:
        peaks = np.array([s[0] for s in session])
        snrs = np.array([s[2] for s in session if s[2] is not None])
        print(f"This session: median peak {np.median(peaks):.1f} dBFS", end="")
        if snrs.size:
            print(f", median SNR {np.median(snrs):.0f} dB")
        else:
            print()
        low = int((peaks < LOW_LEVEL_DBFS).sum())
        if low > len(session) // 2:
            print(f"  {low}/{len(session)} samples were below {LOW_LEVEL_DBFS:.0f} dBFS. "
                  "Raise the input gain before recording more -")
            print("  every sample in the set shares whatever level you record it at.")


if __name__ == "__main__":
    main()
