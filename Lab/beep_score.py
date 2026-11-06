"""
Did the app hear these pulses? For a list of pulse times: pulses with an app event 0 to 0.5 s after, against the same count at shifted control times. Only pulses while the app was listening count: from 10 s after the namer loaded to the last line of its console. With a run folder, also whether the loopback recording has a 1 kHz tone there.
Usage: beep_score.py <pulses csv> <console> [run]
"""
import datetime, json, sys
import numpy as np
pulses = [datetime.datetime.fromisoformat(x.strip()) for x in open(sys.argv[1]) if x.strip()]
ev, ready, last = [], None, None
for line in open(sys.argv[2], encoding="utf-8"):
    try:
        ts = datetime.datetime.fromisoformat(line.split(" ")[0]).timestamp()
    except ValueError:
        continue
    last = ts
    if "distance probe on" in line or "distance probe off" in line:
        ready = ts
    if "[AssistZ] event" in line:
        ev.append(ts)
b = np.array([x.timestamp() for x in pulses])
e = np.array(ev) if ev else np.array([0.0])
if ready is not None and last is not None:
    b = b[(b >= ready + 10) & (b <= last)]
print(f"{len(b)} pulses while listening, {len(ev)} app events")
if len(b) == 0:
    sys.exit("no pulses while the app was listening")
for s in (0.0, -2.0, -1.0, 1.0, 2.0, 3.0):
    label = "THE PULSES" if s == 0 else f"control {s:+.1f} s"
    print(f"  {label:<14} {sum(np.any((e - t - s >= 0) & (e - t - s <= 0.5)) for t in b)}/{len(b)} have an event 0 to 0.5 s after")
if len(sys.argv) > 3:
    run = sys.argv[3].replace("\\", "/")
    pcm = np.load(run + "/audio.npy", mmap_mode="r")
    meta = json.load(open(run + "/audio.json"))
    st = datetime.datetime.fromisoformat(meta["start_utc"]).timestamp()
    rate = meta["rate"]
    heard = 0
    for t in b:
        i = int((t - st + 0.1) * rate)
        if 0 <= i and i + 4800 < len(pcm):
            x = pcm[i:i + 4800, 0].astype(np.float64)
            spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
            k = int(1000 * len(x) / rate)
            heard += spec[k - 3:k + 4].max() > 20 * np.median(spec)
    print(f"loopback recording has a 1 kHz tone at {heard}/{len(b)} pulses (so they did play)")
