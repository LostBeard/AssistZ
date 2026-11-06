"""
Can sound name the gun? All recordings with a logged weapon, each run held out. Ten guns, class-balanced multinomial on the standardised embedding. By gun and by family, near (under 120 m) and far. With confidence: how often it is right when it is sure. Usage: weapon_cv.py
"""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports
R = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_20261105_"
RUNS = {"055044": "_far", "060740": "_far", "062337": "_far", "063920": "_far", "071655": "_f1", "073259": "_f1", "150229": "_f2",
        "194158": "_w", "195820": "_w", "201808": "_w", "203409": "_w", "205010": "_w", "210617": "_w",
        "043053": "_dist", "044227": "_dist", "051052": "_ship", "052232": "_ship", "074844": "_f1", "080020": "_f1", "092756": "_ctl", "151838": "_f2"}
FAMILY = {"SVD": "sniper", "Mosin9130": "sniper", "Winchester70": "sniper", "AKM": "assault", "AugShort": "assault", "M4A1": "assault", "FAL": "assault", "MP5K": "smg", "CZ75": "pistol", "FNX45": "pistol"}
rows = []
for name, tag in RUNS.items():
    path = f"{R}{name}/replay{tag}_s0.jsonl"
    if not os.path.exists(path):
        print("missing", name, tag)
        continue
    reports, _ = calibrated_reports(R + name)
    ev = [json.loads(x) for x in open(path, encoding="utf-8")]
    for r in reports:
        if r[3] > 2000 or r[2] not in FAMILY:
            continue
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        if near:
            e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0]))
            if "embed" in e:
                rows.append(dict(run=name, w=r[2], d=r[3], x=np.array(e["embed"], dtype=np.float64)))
GUNS = sorted(FAMILY)
print(f"{len(rows)} shots from {len({r['run'] for r in rows})} runs; " + " ".join(f"{g} {sum(r['w'] == g for r in rows)}" for g in GUNS))


def fit(rs):
    X = np.stack([r["x"] for r in rs])
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = (X - mean) / std
    y = np.array([GUNS.index(r["w"]) for r in rs])
    K, n = len(GUNS), len(rs)
    count = np.bincount(y, minlength=K)
    w = 1.0 / (K * np.maximum(count[y], 1))
    A, B = np.zeros((K, Z.shape[1])), np.zeros(K)
    Y = np.eye(K)[y]
    for _ in range(800):
        z = Z @ A.T + B
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g = (p - Y) * w[:, None]
        A -= 0.5 * (g.T @ Z + A / n)
        B -= 0.5 * g.sum(axis=0)

    def predict(q):
        s = A @ ((q - mean) / std) + B
        s -= s.max()
        p = np.exp(s)
        return p / p.sum()
    return predict


for run in sorted({r["run"] for r in rows}):
    train = [r for r in rows if r["run"] != run]
    # A gun only in this run cannot be learnt without it; those need the other run.
    f = fit(train)
    for r in rows:
        if r["run"] == run:
            r["p"] = f(r["x"])
rows = [r for r in rows if "p" in r]
for r in rows:
    r["got"] = GUNS[int(np.argmax(r["p"]))]
    r["conf"] = float(np.max(r["p"]))
    fam = collections.Counter()
    for g, v in zip(GUNS, r["p"]):
        fam[FAMILY[g]] += v
    r["famgot"], r["famconf"] = fam.most_common(1)[0]
print("\nGun, each run held out: right / shots, near | far; what it was called")
for g in GUNS:
    n = [r for r in rows if r["w"] == g and r["d"] <= 120]
    f = [r for r in rows if r["w"] == g and r["d"] > 120]
    print(f"  {g:<13} {sum(r['got'] == g for r in n):>4}/{len(n):<4}| {sum(r['got'] == g for r in f):>4}/{len(f):<4}  {dict(collections.Counter(r['got'] for r in n + f).most_common(4))}")
print(f"  all: {sum(r['got'] == r['w'] for r in rows)}/{len(rows)} ({100 * sum(r['got'] == r['w'] for r in rows) / len(rows):.1f}%)")
print("\nFamily, held out:")
for fm in sorted(set(FAMILY.values())):
    g = [r for r in rows if FAMILY[r["w"]] == fm]
    print(f"  {fm:<8} {sum(r['famgot'] == fm for r in g)}/{len(g)}  {dict(collections.Counter(r['famgot'] for r in g).most_common(4))}")
print(f"  all: {sum(r['famgot'] == FAMILY[r['w']] for r in rows)}/{len(rows)} ({100 * sum(r['famgot'] == FAMILY[r['w']] for r in rows) / len(rows):.1f}%)")
print("\nWhen it is sure: gun name")
for bar in (0.5, 0.7, 0.8, 0.9, 0.95):
    k = [r for r in rows if r["conf"] >= bar]
    print(f"  confidence {bar}: says it on {100 * len(k) / len(rows):.0f}% of shots, right {100 * sum(r['got'] == r['w'] for r in k) / max(len(k), 1):.1f}%")
print("\nEach gun at bar 0.8: says it, and when it says it, right")
for g in GUNS:
    said = [r for r in rows if r["got"] == g and r["conf"] >= 0.8]
    have = [r for r in rows if r["w"] == g]
    print(f"  {g:<13} of its shots named {sum(r['got'] == g and r['conf'] >= 0.8 for r in have)}/{len(have)}; when said, right {sum(r['w'] == g for r in said)}/{len(said)}")
print("\nBy range, gun right / family right:")
for lo, hi in ((0, 60), (60, 120), (120, 250), (250, 450), (450, 2000)):
    g = [r for r in rows if lo <= r["d"] < hi]
    if g:
        print(f"  {lo}-{hi} m: {sum(r['got'] == r['w'] for r in g)}/{len(g)} ({100 * sum(r['got'] == r['w'] for r in g) / len(g):.0f}%), family {100 * sum(r['famgot'] == FAMILY[r['w']] for r in g) / len(g):.0f}%")
