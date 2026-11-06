"""
Is the crack on or off rather than louder or quieter? 1) Are the correlations two groups? 2) How does the chance of "on" change with pass distance? 3) Does the 12-20 kHz peak of analyze_flyby.py follow
pass distance once on/off is known, and does it go away when the crack is removed? 4) Does left/right level tell which side the bullet passed?
Usage: python crack_onoff.py <samples dir> <run> [...]
"""
import csv, json, os, sys
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from flyby_match2 import RATE, jobs
from crack_locate import crack_template

samples, runs = sys.argv[1], sys.argv[2:]
lab = r"D:\users\tj\Projects\AssistZ\AssistZ\Lab"
found = {(r["run"], r["i"]): r for r in json.load(open(os.path.join(samples, "flyby_match2.json")))}
HIGH = butter(6, 12000, btype="highpass", fs=RATE, output="sos")
rows = []
for run in runs:
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    client = list(csv.DictReader(open(run + "/aishots_client.csv")))
    impacts = list(csv.DictReader(open(run + "/impacts_client.csv")))
    owner = {}
    for x in impacts:
        t = int(x["game_ms"])
        before = [j for j, c in enumerate(client) if int(c["game_ms"]) <= t <= int(c["game_ms"]) + 2000]
        if before:
            owner.setdefault(max(before, key=lambda j: int(client[j]["game_ms"])), x)
    p = pose[-1]
    head = np.array([float(p["x"]), float(p["y"]) + 1.6, float(p["z"])])
    yaw = np.radians(float(p["ori0"] if "ori0" in p else list(p.values())[5]))
    fwd = np.array([np.sin(yaw), 0.0, np.cos(yaw)])
    right = np.array([np.cos(yaw), 0.0, -np.sin(yaw)])
    for chans, wins, meta in jobs(run, lab):
        r = found[(meta["run"], meta["i"])]
        chans = [c.astype(np.float64) for c in chans]
        y, a = crack_template(samples, r["crack_sample"], r["crack_pitch"])
        pos = wins["crack"][0] + r["crack_k"]
        c = client[meta["i"]]
        q = owner[meta["i"]]
        m = np.array([float(c[k]) for k in ("mx", "my", "mz")])
        e = np.array([float(q[k]) for k in ("x", "y", "z")])
        seg = e - m
        t = np.clip(np.dot(head - m, seg) / np.dot(seg, seg), 0, 1)
        side = float(np.dot(m + t * seg - head, right))
        s0 = wins["search"][0]
        s1 = wins["search"][0] + int(0.095 * RATE)   # arrival -60..+35 ms: holds the peak analyze_flyby.py measured
        c0 = wins["control"][0]
        c1 = c0 + (s1 - s0)
        high = lambda sig, u, v: 20 * np.log10(np.sqrt((sosfiltfilt(HIGH, sig)[u:v] ** 2).max()) + 1e-9)
        base = max(high(ch, c0, c1) for ch in chans)
        before = max(high(ch, s0, s1) for ch in chans) - base
        row = dict(run=meta["run"], i=meta["i"], passed=meta["passed"], d=meta["d"], ncc=r["crack"], pitch=r["crack_pitch"], high=before, side=side)
        on = r["crack"] >= 0.35 and 0.85 <= r["crack_pitch"] <= 1.18
        if on and pos + len(y) <= len(chans[0]):
            nrm2 = float((y ** 2).sum())
            g = [float((ch[pos: pos + len(y)] * y).sum()) / nrm2 for ch in chans]
            row.update(gain_l=g[0], gain_r=g[1])
            for n, ch in enumerate(chans):
                ch[pos: pos + len(y)] -= g[n] * y
            row["high_after"] = max(high(ch, s0, s1) for ch in chans) - base
        row["on"] = bool(on)
        rows.append(row)

v = np.array([r["ncc"] for r in rows])
print(f"{len(rows)} shots. Best crack correlation, count per 0.05 bin from 0.05:")
print("  " + " ".join(f"{lo:.2f}:{((v >= lo) & (v < lo + 0.05)).sum()}" for lo in np.arange(0.05, 0.90, 0.05)))
in_range = [r["ncc"] for r in rows if 0.85 <= r["pitch"] <= 1.18]
out_range = [r["ncc"] for r in rows if not 0.85 <= r["pitch"] <= 1.18]
print(f"  best match at a pitch near 1: {len(in_range)} shots, median {np.median(in_range):.3f}, min {min(in_range):.3f}; elsewhere (a search landing on noise): {len(out_range)}, median {np.median(out_range):.3f}, max {max(out_range):.3f}")
print("\nChance the crack is on, by pass distance:")
for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 12), (12, 18), (18, 25), (25, 35), (35, 1e9)):
    g = [r for r in rows if lo <= r["passed"] < hi]
    if g:
        on = sum(r["on"] for r in g)
        print(f"  {str(lo) + '-' + (str(hi) if hi < 1e9 else 'inf'):<8} {on:3d}/{len(g):<3d} = {on / len(g) * 100:3.0f}%   shooter median {np.median([r['d'] for r in g]):.0f} m")
a, b, c_, d_ = (sum(r["on"] for r in rows if r["passed"] < 8), sum(not r["on"] for r in rows if r["passed"] < 8), sum(r["on"] for r in rows if r["passed"] >= 15), sum(not r["on"] for r in rows if r["passed"] >= 15))
print(f"  under 8 m {a}/{a + b} vs 15 m or more {c_}/{c_ + d_}: Fisher p = {fisher_exact([[a, b], [c_, d_]])[1]:.2e}")
print("  by run, on / shots: " + str({run: f"{sum(r['on'] for r in rows if r['run'] == run)}/{sum(r['run'] == run for r in rows)}" for run in sorted({r['run'] for r in rows})}))
print("\nThe 12-20 kHz signal (dB over the same shot's earlier window):")
for name, sel in (("under 5 m, crack on", lambda r: r["passed"] < 5 and r["on"]), ("under 5 m, crack off", lambda r: r["passed"] < 5 and not r["on"]), ("5-15 m, crack on", lambda r: 5 <= r["passed"] < 15 and r["on"]), ("5-15 m, crack off", lambda r: 5 <= r["passed"] < 15 and not r["on"]), ("15 m or more, crack on", lambda r: r["passed"] >= 15 and r["on"]), ("15 m or more, crack off", lambda r: r["passed"] >= 15 and not r["on"])):
    g = [r for r in rows if sel(r)]
    if g:
        after = [r["high_after"] for r in g if "high_after" in r]
        print(f"  {name:<26} n={len(g):3d}  median {np.median([r['high'] for r in g]):5.1f} dB" + (f"; with the crack subtracted {np.median(after):5.1f} dB" if after else ""))
on_rows = [r for r in rows if r["on"]]
off_rows = [r for r in rows if not r["on"]]
print(f"  on vs off, all distances: {np.median([r['high'] for r in on_rows]):.1f} vs {np.median([r['high'] for r in off_rows]):.1f} dB, p = {mannwhitneyu([r['high'] for r in on_rows], [r['high'] for r in off_rows], alternative='greater').pvalue:.2e}")
print(f"  crack on only, Spearman 12-20 kHz vs pass distance: {spearmanr([r['high'] for r in on_rows], [r['passed'] for r in on_rows]).statistic:+.3f}")
print("\nLeft/right when the crack is on (level difference R - L, dB, against the side the bullet passed, + = right):")
g = [r for r in on_rows if "gain_l" in r and abs(r["side"]) > 0.5]
ild = np.array([20 * np.log10((abs(r["gain_r"]) + 1e-6) / (abs(r["gain_l"]) + 1e-6)) for r in g])
side = np.array([r["side"] for r in g])
print(f"  n={len(g)}; passed right {int((side > 0).sum())}, left {int((side < 0).sum())}; median ILD {np.median(ild[side > 0]) if (side > 0).any() else 0:+.1f} dB when right, {np.median(ild[side < 0]) if (side < 0).any() else 0:+.1f} dB when left; Spearman ILD vs side {spearmanr(ild, side).statistic:+.3f}; sign agrees {np.mean(np.sign(ild) == np.sign(side)) * 100:.0f}%")
