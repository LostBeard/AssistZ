"""
How far away was the shot? First honest look, from recordings already made, each run left out in turn. True distance comes from the shot logs. Cues: level alone; level per weapon; a linear fit on CLAP's embedding; and the two together. Scored as median error in metres against always answering the median distance. Not gated.
"""
import csv, glob, json, os, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

ROOT = "D:/users/tj/Projects/AssistZ/Lab/runs/"
rows = []
for path in sorted(glob.glob(ROOT + "sniper_*/shots_ai.csv")):
    run = os.path.dirname(path).replace("\\", "/")
    name = os.path.basename(run)
    tags = [t for t in ("_dir", "_rev", "_base", "_vote", "_emb", "_app") if os.path.exists(f"{run}/replay{t}_s0.jsonl")]
    if not tags:
        continue
    first = open(f"{run}/replay{tags[0]}_s0.jsonl", encoding="utf-8").readline()
    if not first or "embed" not in json.loads(first):
        continue
    reports, _ = calibrated_reports(run)
    ev = [json.loads(x) for x in open(f"{run}/replay{tags[0]}_s0.jsonl", encoding="utf-8")]
    a = np.load(run + "/audio.npy", mmap_mode="r")
    weather = "clear"
    if os.path.exists(run + "/world.csv"):
        evs = [e["event"] for e in csv.DictReader(open(run + "/world.csv"))]
        weather = "storm" if "storm" in evs else ("rain" if "rain" in evs else "clear")
    for r in reports:
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        if not near:
            continue
        e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0]))
        i = e["sample"]
        seg = np.array(a[i: i + 9600], dtype=np.float64).mean(axis=1)
        if len(seg) < 9600:
            continue
        sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        fr = np.fft.rfftfreq(len(seg), 1 / 48000)
        total = sp.sum() + 1e-12
        bands = [10 * np.log10(sp[(fr >= lo) & (fr < hi)].sum() / total + 1e-9) for lo, hi in ((0, 200), (200, 800), (800, 3000), (3000, 8000), (8000, 24000))]
        rows.append(dict(run=name, weapon=r[2], dist=float(r[3]), level=float(e["level"]), weather=weather, x=np.array(e["embed"], dtype=np.float32), bands=bands))
print(len(rows), "events from", len({r["run"] for r in rows}), "runs")
print("weapons:", {w: sum(r["weapon"] == w for r in rows) for w in sorted({r["weapon"] for r in rows})})
print("weather:", {w: sum(r["weather"] == w for r in rows) for w in sorted({r["weather"] for r in rows})})
d = np.array([r["dist"] for r in rows])
print("true distance percentiles 5, 25, 50, 75, 95:", [round(float(x)) for x in np.percentile(d, (5, 25, 50, 75, 95))], "max", round(float(d.max())))
for c in (25, 40, 50, 65, 75, 90, 100, 150):
    g = [r["level"] for r in rows if abs(r["dist"] - c) < 8]
    if len(g) > 5:
        print(f"  about {c} m: {len(g)} events, level median {np.median(g):.1f} dB, p10-p90 {np.percentile(g, 10):.1f} to {np.percentile(g, 90):.1f}")
y = np.log2(d)
runs = np.array([r["run"] for r in rows])
W = sorted({r["weapon"] for r in rows})


def ridge(X, y, lam):
    mx, sx = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = (X - mx) / sx
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return lambda q: ((q - mx) / sx) @ w + y.mean()


def loro(X, title, lam=1.0, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    pred = np.full(len(y), np.nan)
    for run in sorted(set(runs[m])):
        te = m & (runs == run)
        tr = m & (runs != run)
        if tr.sum() < 20:
            continue
        pred[te] = ridge(X[tr], y[tr], lam)(X[te])
    ok = ~np.isnan(pred)
    e = np.abs(2 ** pred[ok] - d[ok])
    base = np.abs(np.median(d[ok]) - d[ok])
    r = np.corrcoef(pred[ok], y[ok])[0, 1]
    print(f"  {title:<34} median error {np.median(e):5.1f} m (always the median {np.median(base):5.1f}); within 25% {100 * (e <= 0.25 * d[ok]).mean():4.1f}% (median {100 * (base <= 0.25 * d[ok]).mean():4.1f}%); correlation {r:.2f}; {ok.sum()} events")


level = np.array([[r["level"]] for r in rows], dtype=np.float64)
one = np.array([[r["weapon"] == w for w in W] for r in rows], dtype=np.float64)
emb = np.stack([r["x"] for r in rows]).astype(np.float64)
bands = np.array([r["bands"] for r in rows], dtype=np.float64)
print("\nEach run left out in turn, all weather:")
loro(level, "level")
loro(np.hstack([level, one]), "level + true weapon")
loro(np.hstack([level, bands]), "level + spectrum")
loro(np.hstack([level, bands, one]), "level + spectrum + true weapon")
for lam in (10.0, 100.0, 1000.0, 5000.0):
    loro(emb, f"CLAP embedding, ridge {lam:g}", lam)
    loro(np.hstack([emb, level, one]), f"embedding + level + weapon, {lam:g}", lam)
print("\nClear weather only:")
clear = np.array([r["weather"] == "clear" for r in rows])
loro(level, "level", mask=clear)
loro(np.hstack([level, one]), "level + true weapon", mask=clear)
loro(np.hstack([emb, level, one]), "embedding + level + weapon, 1000", 1000.0, mask=clear)
print("\nPer burst, median of its events (what a ping would show), embedding + level + weapon, 1000:")
pred = np.full(len(y), np.nan)
X = np.hstack([emb, level, one])
for run in sorted(set(runs)):
    te = runs == run
    if (~te).sum() >= 20:
        pred[te] = ridge(X[~te], y[~te], 1000.0)(X[te])
ok = ~np.isnan(pred)
groups = {}
for i, r in enumerate(rows):
    if ok[i]:
        groups.setdefault((r["run"], r["weapon"], round(r["dist"] / 5)), []).append(i)
p = np.array([2 ** np.median(pred[v]) for v in groups.values()])
t = np.array([d[v[0]] for v in groups.values()])
print(f"  {len(p)} groups; median error {np.median(np.abs(p - t)):.1f} m (always the median {np.median(np.abs(np.median(t) - t)):.1f}); within 25% {100 * (np.abs(p - t) <= 0.25 * t).mean():.1f}%; correlation {np.corrcoef(np.log2(p), np.log2(t))[0, 1]:.2f}")
