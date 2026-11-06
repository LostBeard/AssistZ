"""First honest numbers, from recordings already on disk: direction error against the true bearing; weapon and distance band from the CLAP embedding, each run left out in turn. Exploratory, not gated."""
import csv, glob, json, math, os, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

ROOT = "D:/users/tj/Projects/AssistZ/Lab/runs/"
rows = []
report = []
for path in sorted(glob.glob(ROOT + "sniper_*/shots_ai.csv")):
    run = os.path.dirname(path).replace("\\", "/")
    name = os.path.basename(run)
    tags = sorted(glob.glob(run + "/replay*_s0.jsonl"), key=lambda p: ("_rev" not in p, "_app" not in p, "_emb" not in p, "_vote" not in p, p))
    tags = [t for t in tags if json.loads(open(t, encoding="utf-8").readline() or "{}").get("embed")]
    if not tags:
        continue
    shots = list(csv.DictReader(open(path)))
    if not shots:
        continue
    if not os.path.exists(run + "/aishots_server.csv"):
        continue
    reports, shift = calibrated_reports(run)
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    pg = np.array([int(p["game_ms"]) for p in pose])
    server = list(csv.DictReader(open(run + "/aishots_server.csv")))
    sg = np.array([int(s["game_ms"]) for s in server])
    ev = [json.loads(x) for x in open(tags[0], encoding="utf-8")]
    used = 0
    for r, s in zip(reports, shots):
        g = int(s["game_ms"])
        k = int(np.argmin(np.abs(sg - g)))
        if abs(sg[k] - g) > 1500:
            continue
        p = pose[int(np.argmin(np.abs(pg - g)))]
        dx = float(server[k]["mx"]) - float(p["x"])
        dz = float(server[k]["mz"]) - float(p["z"])
        bearing = math.degrees(math.atan2(dx, dz))
        rel = (bearing - float(p["yaw"]) + 180) % 360 - 180
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        if not near:
            rows.append(dict(run=name, heard=False, weapon=r[2], dist=r[3], rel=rel))
            continue
        e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0]))
        rows.append(dict(run=name, heard=True, weapon=r[2], dist=r[3], rel=rel, az=float(e["az"]), level=float(e["level"]), x=np.array(e["embed"]), label=e["label"]))
        used += 1
    report.append((name, os.path.basename(tags[0]), len(shots), used, round(shift, 3)))
print(f"{len(report)} recordings with logged shots and embeddings")
for r in report:
    print("  ", r)
heard = [r for r in rows if r["heard"]]
print(f"\nDETECTION: {len(heard)}/{len(rows)} logged shots have an event ({100 * len(heard) / max(len(rows), 1):.0f}%)")
print("\nDIRECTION. Stereo cannot tell front from back, so the true bearing is folded to the front half before comparing.")
fold = lambda a: (180 - a if a > 90 else (-180 - a if a < -90 else a))
err = np.array([r["az"] - fold(r["rel"]) for r in heard])
print(f"  folded error: median |e| {np.median(np.abs(err)):.0f} deg, p90 {np.percentile(np.abs(err), 90):.0f}, bias {np.median(err):+.0f}, correlation {np.corrcoef([r['az'] for r in heard], [fold(r['rel']) for r in heard])[0, 1]:.2f}, sign flipped correlation {np.corrcoef([-r['az'] for r in heard], [fold(r['rel']) for r in heard])[0, 1]:.2f}")
side = [(np.sign(r["az"]), np.sign(r["rel"])) for r in heard if abs(r["az"]) > 5 and 15 < abs(fold(r["rel"]))]
print(f"  left or right right: {sum(a == b for a, b in side)}/{len(side)} ({100 * sum(a == b for a, b in side) / max(len(side), 1):.0f}%), where the true direction is at least 15 deg off centre")
front = sum(abs(r["rel"]) <= 90 for r in heard)
print(f"  true bearings in front {front}, behind {len(heard) - front}; the app has no front or back cue")
print("  by run: az vs truth correlation " + ", ".join(f"{n[-6:]} {np.corrcoef([r['az'] for r in heard if r['run'] == n], [fold(r['rel']) for r in heard if r['run'] == n])[0, 1]:.2f}" for n in sorted({r['run'] for r in heard}) if sum(r['run'] == n for r in heard) > 8 and np.std([fold(r['rel']) for r in heard if r['run'] == n]) > 1))
print(f"  spread of true direction: std {np.std([fold(r['rel']) for r in heard]):.0f} deg over {len(set(round(fold(r['rel']) / 10) for r in heard))} 10-degree bins")


def fit(X, y, K, steps=500, lr=0.5, l2=1.0):
    n, dim = X.shape
    count = np.bincount(y, minlength=K)
    w = 1.0 / (K * np.maximum(count[y], 1))
    A, B = np.zeros((K, dim)), np.zeros(K)
    Y = np.eye(K)[y]
    for _ in range(steps):
        z = X @ A.T + B
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        r = (p - Y) * w[:, None]
        A -= lr * (r.T @ X + l2 / n * A)
        B -= lr * r.sum(axis=0)
    return A, B


def loro(values, title):
    d = [r for r in heard if r.get(values) is not None]
    names = sorted({r[values] for r in d})
    if len(names) < 2:
        print(title, "one class only")
        return
    y = np.array([names.index(r[values]) for r in d])
    X = np.stack([r["x"] for r in d])
    runs = np.array([r["run"] for r in d])
    conf = np.zeros((len(names), len(names)), dtype=int)
    scored_runs = 0
    for run in sorted(set(runs)):
        te = runs == run
        tr = ~te
        have = set(y[tr].tolist())
        if len(have) < 2:
            continue
        A, B = fit(X[tr], y[tr], len(names))
        pred = np.argmax(X[te] @ A.T + B, axis=1)
        for a, b in zip(y[te], pred):
            conf[a, b] += 1
        scored_runs += 1
    total = conf.sum()
    print(f"\n{title}: {np.trace(conf)}/{total} right ({100 * np.trace(conf) / max(total, 1):.0f}%), chance about {100 * np.bincount(y).max() / len(y):.0f}% always saying the commonest; {scored_runs} runs left out in turn")
    print("  truth\\said  " + " ".join(f"{n[:9]:>9}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n[:11]:<11} " + " ".join(f"{v:>9d}" for v in conf[i]) + f"   {100 * conf[i, i] / max(conf[i].sum(), 1):.0f}%")


for r in heard:
    r["band"] = "under 60 m" if r["dist"] < 60 else ("60-120 m" if r["dist"] < 120 else "over 120 m")
print("\nshots by weapon:", {w: sum(r["weapon"] == w for r in heard) for w in sorted({r["weapon"] for r in heard})})
print("shots by true distance (nearest 25 m):", dict(sorted({int(round(k / 25) * 25): sum(round(r["dist"] / 25) * 25 == round(k / 25) * 25 for r in heard) for k in [r["dist"] for r in heard]}.items())))
loro("weapon", "WEAPON NAME from the embedding")
loro("band", "DISTANCE BAND from the embedding")
