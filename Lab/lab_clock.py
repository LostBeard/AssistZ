"""
A lab run's shot log on the recording's own clock.

The pose log's UTC column is whole seconds, so the game -> recording clock is good to a few hundred ms only. It is calibrated
on the data: the single shift that puts the most broadband energy at every shot's due report time (the method of
analyze_crack_timing.py). The search spans +-SEARCH_SECONDS: at +-0.5 s recording sniper_20261103_190530 read -35 ms, a false
peak, with its real peak near -2.4 s (mean level at the due times -28.4 dB).
"""
import csv, datetime, json, os
import numpy as np

HOP = 240
SEARCH_SECONDS = 5.0


def envelope_db(pcm):
    mono = pcm.mean(axis=1)
    n = len(mono) // HOP
    frames = mono[: n * HOP].reshape(n, HOP)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)


def calibrated_reports(run):
    """([(report time s, shooter, weapon, distance m, miss m)], the shift applied in s)."""
    if not os.path.exists(run + "/shots_ai.csv"):
        return [], 0.0   # a recording with no gunfire at all (lab mode 10)
    pcm = np.load(run + "/audio.npy").astype(np.float64)
    meta = json.load(open(run + "/audio.json"))
    rate = meta["rate"]
    st = datetime.datetime.fromisoformat(meta["start_utc"])
    start_s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    g = np.array([int(r["game_ms"]) / 1000 for r in pose])
    u = np.array([sum(int(x) * k for x, k in zip(r["utc"].split(":"), (3600, 60, 1))) for r in pose], dtype=float)
    offset = np.median(u + 0.5 - g)
    shots = list(csv.DictReader(open(run + "/shots_ai.csv")))
    dues = [int(x["game_ms"]) / 1000 + offset - start_s + float(x["distance"]) / 330.0 for x in shots]
    dt = HOP / rate
    full = envelope_db(pcm)
    shifts = np.arange(-SEARCH_SECONDS, SEARCH_SECONDS, dt)
    score = [np.mean([full[int((t + sh) / dt)] for t in dues if 0 <= int((t + sh) / dt) < len(full)]) for sh in shifts]
    shift = float(shifts[int(np.argmax(score))])
    return ([(t + shift, x["shooter"].replace("eAI_Survivor", ""), x["weapon"], float(x["distance"]), float(x["miss"]))
             for t, x in zip(dues, shots)], shift)
