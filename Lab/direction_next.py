"""From what is already recorded: (1) can a curve fixed on one run correct the angle on the other? (2) does anything in the sound separate front from back? (3) are 'missed shots' missed bursts, or a limit on events per second? Exploratory, not gated."""
import csv, json, math, os, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

ROOT = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_"
A, B = "20261105_000715", "20261105_001851"
fold = lambda a: (180 - a if a > 90 else (-180 - a if a < -90 else a))


def load(name, tag="_dir"):
    run = ROOT + name
    reports, shift = calibrated_reports(run)
    shots = list(csv.DictReader(open(run + "/shots_ai.csv")))
    server = list(csv.DictReader(open(run + "/aishots_server.csv")))
    sg = np.array([int(s["game_ms"]) for s in server])
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    pg = np.array([int(p["game_ms"]) for p in pose])
    ev = [json.loads(x) for x in open(f"{run}/replay{tag}_s0.jsonl", encoding="utf-8")]
    a = np.load(run + "/audio.npy", mmap_mode="r")
    out = []
    for r, s in zip(reports, shots):
        g = int(s["game_ms"])
        k = int(np.argmin(np.abs(sg - g)))
        if abs(sg[k] - g) > 1500:
            continue
        p = pose[int(np.argmin(np.abs(pg - g)))]
        dx, dz = float(server[k]["mx"]) - float(p["x"]), float(server[k]["mz"]) - float(p["z"])
        rel = (math.degrees(math.atan2(dx, dz)) - float(p["yaw"]) + 180) % 360 - 180
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        if not near:
            continue
        e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0]))
        i = e["sample"]
        seg = np.array(a[i: i + 4800], dtype=np.float64).mean(axis=1)
        sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        fr = np.fft.rfftfreq(len(seg), 1 / 48000)
        high = 10 * np.log10(sp[fr >= 4000].sum() / (sp.sum() + 1e-12) + 1e-12)
        out.append(dict(rel=rel, az=float(e["az"]), level=float(e["level"]), high=high, x=np.array(e["embed"])))
    return out, reports, ev


da, ra, ea = load(A)
db, rb, eb = load(B)


def curve(train):
    bins = np.arange(-90, 91, 15)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        g = [fold(r["rel"]) for r in train if lo <= r["az"] < hi]
        if len(g) >= 5:
            xs.append((lo + hi) / 2)
            ys.append(float(np.median(g)))
    ys = np.maximum.accumulate(ys)
    return lambda az: float(np.interp(az, xs, ys))


def score(d, f, title):
    err = np.array([f(r["az"]) - fold(r["rel"]) for r in d])
    row = []
    worst = 0
    for b in range(-90, 91, 30):
        g = [abs(f(r["az"]) - fold(r["rel"])) for r in d if abs(fold(r["rel"]) - b) <= 15]
        if len(g) >= 5:
            row.append(f"{b}:{np.median(g):.0f}")
            worst = max(worst, float(np.median(g)))
    print(f"  {title:<34} median {np.median(np.abs(err)):5.1f}, p90 {np.percentile(np.abs(err), 90):5.1f}; by true folded angle {' '.join(row)}; worst bearing median {worst:.0f}")


print("(1) ANGLE, in the app today, and with a curve fixed on the other run")
score(db, lambda a: a, f"{B[-6:]} raw")
score(db, curve(da), f"{B[-6:]} curve from {A[-6:]}")
score(da, lambda a: a, f"{A[-6:]} raw")
score(da, curve(db), f"{A[-6:]} curve from {B[-6:]}")
print("  mapping from run 1: az -> true folded:", {a: round(curve(da)(a)) for a in (-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75)})
print("\n(2) FRONT OR BACK at mirrored angles, events 20-70 deg off the axis")


def fb(d):
    return [r for r in d if 20 <= abs(fold(r["rel"])) <= 70]


for name, d in ((A[-6:], da), (B[-6:], db)):
    g = fb(d)
    f = [r for r in g if abs(r["rel"]) <= 90]
    k = [r for r in g if abs(r["rel"]) > 90]
    print(f"  {name}: front {len(f)}, level {np.mean([r['level'] for r in f]):.1f} dB, above 4 kHz {np.mean([r['high'] for r in f]):.1f} dB; back {len(k)}, level {np.mean([r['level'] for r in k]):.1f} dB, above 4 kHz {np.mean([r['high'] for r in k]):.1f} dB")


def fit(X, y, steps=500, lr=0.5, l2=1.0):
    n, dim = X.shape
    w = 1 / (2 * np.bincount(y, minlength=2)[y])
    W, b = np.zeros(dim), 0.0
    for _ in range(steps):
        z = X @ W + b
        p = 1 / (1 + np.exp(-z))
        r = (p - y) * w
        W -= lr * (r @ X + l2 / n * W)
        b -= lr * r.sum()
    return W, b


for tr, te, title in ((da, db, f"train {A[-6:]}, test {B[-6:]}"), (db, da, f"train {B[-6:]}, test {A[-6:]}")):
    a, c = fb(tr), fb(te)
    ya = np.array([abs(r["rel"]) > 90 for r in a], dtype=int)
    yc = np.array([abs(r["rel"]) > 90 for r in c], dtype=int)
    W, b0 = fit(np.stack([r["x"] for r in a]), ya)
    pred = (np.stack([r["x"] for r in c]) @ W + b0) > 0
    bal = (pred[yc == 1].mean() + (~pred[yc == 0]).mean()) / 2
    print(f"  embedding probe, {title}: front {int((~pred[yc == 0]).sum())}/{int((yc == 0).sum())}, back {int(pred[yc == 1].sum())}/{int((yc == 1).sum())}, balanced {100 * bal:.0f}% (coin 50%)")
print("\n(3) MISSED SHOTS. A burst is shots under 1 s apart. Heard means at least one event within its span.")
for name, title in ((A, "direction run"), (B, "direction run"), ("20261104_193426", "rain storm"), ("20261104_215004", "rain storm"), ("20261104_194250", "dry storm"), ("20261104_215827", "dry storm"), ("20261104_220700", "rain, no storm"), ("20261104_002833", "clear"), ("20261104_001133", "clear")):
    run = ROOT + name
    reports, _ = calibrated_reports(run)
    tag = next(t for t in ("_dir", "_rev", "_base", "_vote", "_emb") if os.path.exists(f"{run}/replay{t}_s0.jsonl"))
    ev = np.array(sorted(json.loads(x)["sample"] / 48000.0 for x in open(f"{run}/replay{tag}_s0.jsonl", encoding="utf-8")))
    t = np.array(sorted(r[0] for r in reports))
    groups = np.split(t, np.where(np.diff(t) >= 1.0)[0] + 1)
    heard = sum(((ev >= g[0] - 0.05) & (ev <= g[-1] + 0.15)).any() for g in groups)
    per = sum(((ev >= x - 0.05) & (ev <= x + 0.15)).any() for x in t)
    gaps = np.diff(t)
    print(f"  {name[-6:]} {title:<15} shots with an event {per}/{len(t)} ({100 * per / len(t):.0f}%); bursts heard {heard}/{len(groups)} ({100 * heard / len(groups):.0f}%); median shot gap {np.median(gaps):.2f} s; event gaps p10 {np.percentile(np.diff(ev), 10):.2f} s")
