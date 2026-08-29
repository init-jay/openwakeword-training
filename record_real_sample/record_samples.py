#!/usr/bin/env python3
"""
Record real voice samples for wake word training.
Creates 16kHz mono WAV files in the repo's my_real_samples/.

Lives in its own directory with its own uv environment: it runs on the host for
microphone access and needs only numpy, whereas the training environment pins torch
and tensorflow. Keeping them apart means recording does not need the trainer's stack.

Capture goes through ffmpeg's avfoundation input, matching test_model.py, so there
is no PyAudio/PortAudio to build - just `ffmpeg` on PATH.

Usage:
    cd record_real_sample
    uv run record_samples.py --list-devices
    uv run record_samples.py --wake-word "hey cal"
    uv run record_samples.py --wake-word "hey cal" --output-dir ../my_real_samples/jay
"""

import argparse
import re
import subprocess
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2                     # 16-bit
DURATION = 2.0  # seconds of usable audio captured after the cue
WARMUP = 0.6    # seconds discarded after opening the device, before the cue

# Samples belong to the repo, not to this tool's directory, so the default output
# path is anchored to the repo root rather than the working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

FULL_SCALE = 32768.0
# Speech should peak somewhere near -12 dBFS.
#
# Be clear about what this does and does not buy. Absolute level turns out not to
# matter to the model at all: a synthetic sweep from -6 to -34 dBFS detected 6/6 at
# every step with a flat median score of ~0.984. What does matter is SNR, which
# degrades detection below about 15 dB. Recording hot is therefore about margin -
# it keeps SNR high and leaves headroom - not about a level the model needs.
#
# The reason to warn at capture rather than fix it later is that SNR is fixed at
# capture: amplifying a quiet clip raises its noise with it.
LOW_LEVEL_DBFS = -18.0
CLIPPING_DBFS = -0.5
LOW_SNR_DB = 20.0


def next_index(output_dir: Path, safe_name: str) -> int:
    """One past the highest existing index for this wake word.

    Numbering from a file COUNT breaks as soon as the sequence has gaps: jay's
    directory held 56 clips numbered up to 0062 (takes had been deleted), so
    count + 1 pointed at 0057, which already existed - and recording would have
    silently overwritten five good samples before reaching free numbers.
    """
    pattern = re.compile(rf"^{re.escape(safe_name)}_(\d+)\.wav$")
    indices = [int(m.group(1)) for f in output_dir.glob("*.wav")
               if (m := pattern.match(f.name))]
    return max(indices, default=0) + 1


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


def list_devices():
    """Print avfoundation audio input devices and their numbers."""
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    ).stderr
    audio = out.split("AVFoundation audio devices:")
    if len(audio) < 2:
        print(out)
        return
    print("Audio input devices (use the number with --device):")
    for line in audio[1].splitlines():
        m = re.search(r"\[(\d+)\]\s+(.*)", line)
        if m:
            print(f"  {m.group(1)}: {m.group(2).strip()}")


def open_stream(device: str) -> subprocess.Popen:
    """Spawn ffmpeg streaming raw 16kHz mono 16-bit PCM to stdout."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "avfoundation", "-i", f":{device}",
        "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
        "-f", "s16le", "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def read_exact(pipe, nbytes: int):
    """Read exactly nbytes from a pipe, or None if the stream ends.

    A pipe read can come back short, so this loops - a partial read here would
    silently shorten the recording.
    """
    buf = b""
    while len(buf) < nbytes:
        chunk = pipe.read(nbytes - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def record_sample(filename: str, device: str):
    """Record one sample, cueing once the stream is live. Returns its levels."""
    proc = open_stream(device)
    if proc.stdout is None:
        raise SystemExit("failed to open ffmpeg pipes")

    try:
        # An input device does not deliver usable audio the instant it opens. Read
        # WARMUP seconds first so the cue below lands on an already-running stream -
        # cueing before the device settles loses the first fraction of a second of
        # speech and clips the word onset (the "h" in "hey"). This audio is not
        # written to the sample, but its second half is kept as a noise-floor
        # reference; the first half can still hold the device settling.
        warmup = read_exact(proc.stdout, int(SAMPLE_RATE * WARMUP) * SAMPLE_WIDTH)
        if warmup is None:
            err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            raise SystemExit(
                "ffmpeg could not open the microphone. If this is the first run, macOS "
                "may be waiting on a microphone permission prompt for your terminal.\n"
                + err)

        print("SPEAK NOW!")
        audio = read_exact(proc.stdout, int(SAMPLE_RATE * DURATION) * SAMPLE_WIDTH)
        if audio is None:
            raise SystemExit("microphone stream ended mid-recording")
    finally:
        proc.kill()
        proc.wait()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio)

    speech = np.frombuffer(audio, dtype=np.int16)
    noise = np.frombuffer(warmup[len(warmup) // 2:], dtype=np.int16)
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
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "my_real_samples"),
                        help="Output directory (default: %(default)s). Use a "
                             "per-speaker subdirectory when several people record.")
    parser.add_argument("--device", default="0", help="avfoundation audio device number")
    parser.add_argument("--list-devices", action="store_true",
                        help="List microphones and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = args.wake_word.replace(" ", "_").lower()

    print("=" * 50)
    print(f"Voice Sample Recorder - \"{args.wake_word}\"")
    print("=" * 50)
    print()
    print("Record at least 20-50 samples for best results.")
    print("Vary your tone, speed, distance from mic, etc.")
    print()
    print("  - Press ENTER to start recording")
    print("  - Capture goes through ffmpeg; --list-devices shows microphones")
    print(f"  - Wait for \"SPEAK NOW!\", then say \"{args.wake_word}\" naturally")
    print("  - Recording lasts 2 seconds; silence is trimmed at training time")
    print("  - Levels are reported after each take; aim for a peak near -12 dBFS")
    print("  - Press 'q' + ENTER to quit")
    print()

    existing = len(list(output_dir.glob("*.wav")))
    index = next_index(output_dir, safe_name)
    print(f"Existing samples: {existing} (next file: {safe_name}_{index:04d}.wav)")
    print()

    session = []
    while True:
        user_input = input(f"[Sample {existing + 1}] Press ENTER to record (q to quit): ")

        if user_input.lower() == 'q':
            break

        print("Get ready...", end=" ", flush=True)
        time.sleep(0.5)

        path = output_dir / f"{safe_name}_{index:04d}.wav"
        if path.exists():                      # never clobber an existing take
            raise SystemExit(f"Refusing to overwrite {path}")
        levels = record_sample(str(path), args.device)

        print(f"Saved: {path}")
        level_report(*levels)
        session.append(levels)
        existing += 1
        index += 1
        print()

    print(f"\nDone! {existing} total samples in {output_dir}/")

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
