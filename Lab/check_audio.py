"""
Is a lab recording fit to analyse? Fails (exit 1) when a run of digital silence lasts over MAX_SILENCE_SECONDS. Game audio has a noise
floor near -55 dB, so silence means the stream stopped: run sniper_20261103_232810 went silent at 32.1 s for 157.9 s, the DayZ server and
client logged a normal run, and its shots read 0.0 dB. Silence from the first sample up to MAX_START_SILENCE_SECONDS is normal: the recorder starts before the stream has sound (2.5 s seen on Nasdara, up to 1.25 s on Chernarus).
Usage: python check_audio.py <run dir>
"""
import json, sys
import numpy as np

MAX_SILENCE_SECONDS = 2.0
MAX_START_SILENCE_SECONDS = 5.0
run = sys.argv[1]
pcm = np.load(run + "/audio.npy")   # not a memmap: mapping a large file on the D: share fails
rate = json.load(open(run + "/audio.json"))["rate"]
zero = (np.asarray(pcm) == 0).all(axis=1)
edges = np.flatnonzero(np.diff(np.concatenate(([0], zero.astype(np.int8), [0]))))
longest, at = 0.0, 0.0
start = 0.0
for a, b in zip(edges[::2], edges[1::2]):
    if a == 0 and (b - a) / rate <= MAX_START_SILENCE_SECONDS:
        start = (b - a) / rate
        continue
    if (b - a) / rate > longest:
        longest, at = (b - a) / rate, a / rate
level = 20 * np.log10(np.sqrt(np.mean(np.asarray(pcm, dtype=np.float64) ** 2)) + 1e-12)
print(f"audio check: {len(pcm) / rate:.0f} s, level {level:.1f} dB, longest digital silence {longest:.2f} s at {at:.1f} s, quiet start {start:.2f} s")
if longest > MAX_SILENCE_SECONDS:
    print(f"AUDIO INVALID: over {MAX_SILENCE_SECONDS:.0f} s of digital silence")
    sys.exit(1)
