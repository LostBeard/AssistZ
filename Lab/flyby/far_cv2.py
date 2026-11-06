"""
Distance in two steps, each far run held out: is it past 120 m (class-balanced logistic on the standardised embedding), then a near ridge or a far-only ridge. Near side scored on the two fresh mode 28 runs that no fit has seen. Then the candidate namer's labels as tag _off, for phase_gates7.py.
Usage (from Lab/flyby): python far_cv2.py <cache.pkl>
"""
import json, os, pickle, sys
import numpy as np
from world_classes import CLASSES, U, gunstorm_rows
from far_cv import FAR, NAMES, STEPS, far_shots, fit, ridge
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

CUT = 120.0


def logistic(X, far):
    mx, sx = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = (X - mx) / sx
    w8 = np.where(far, 0.5 / far.sum(), 0.5 / (~far).sum())
    a, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(800):
        p = 1 / (1 + np.exp(-np.clip(Z @ a + b, -30, 30)))
        r = (p - far) * w8
        a -= 0.5 * (Z.T @ r + a / len(Z))
        b -= 0.5 * r.sum()
    return lambda q: ((np.asarray(q, dtype=np.float64) - mx) / sx) @ a + b


def fresh(run):
    reports, _ = calibrated_reports(U + run)
    ev = [json.loads(x) for x in open(f"{U}{run}/replay_ship_s0.jsonl", encoding="utf-8")]
    out = []
    for r in reports:
        if r[3] > 150:
            continue
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        if near:
            out.append((min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0]))["embed"], r[3], r[2]))
    return out


if __name__ == "__main__":
    base, DX, Dy = pickle.load(open(sys.argv[1], "rb"))
    data = {n: far_shots(n) for n in FAR}
    near0 = ridge(DX[2 ** Dy <= CUT], Dy[2 ** Dy <= CUT])
    fresh_runs = {r: fresh(r) for r in ("20261105_051052", "20261105_052232")}
    print(f"distance rows {len(Dy)}, of them past {CUT:.0f} m: {(2 ** Dy > CUT).sum()}")
    print("\n=== Past 120 m? Each far run held out. Called far / events, by range; weapons in the last column")
    sumfar = {s: [0, 0] for s in STEPS}
    allp, allt, wrong_near = [], [], []
    for n in FAR:
        Xf = np.array([x[0]["embed"] for m in FAR if m != n for x in data[m][1]], dtype=np.float64)
        Tf = np.array([x[3] for m in FAR if m != n for x in data[m][1]])
        X = np.concatenate([DX, Xf])
        far = np.concatenate([2 ** Dy > CUT, Tf > CUT])
        g = logistic(X, far)
        fr = ridge(Xf, np.log2(Tf))
        line = []
        for s in STEPS:
            h = [x for x in data[n][1] if x[1] == s]
            c = sum(g(x[0]["embed"]) > 0 for x in h)
            sumfar[s][0] += c; sumfar[s][1] += len(h)
            line.append(f"{s}:{c}/{len(h)}")
            if h:
                allp += [float(fr(x[0]["embed"])) for x in h]; allt += [x[3] for x in h]
        med = " ".join(f"{s}:{np.median([float(fr(x[0]['embed'])) for x in data[n][1] if x[1] == s]):.0f}" for s in STEPS if any(x[1] == s for x in data[n][1]))
        print(f"  {n[-6:]} {NAMES[n[-6:]]:<10} {' '.join(line)}   far-only ridge: {med}")
        for run, rows in fresh_runs.items():
            cnt = sum(g(r[0]) > 0 for r in rows)
            wrong_near.append((n[-6:], run[-6:], cnt, len(rows)))
    print("  all: " + " ".join(f"{s}:{v[0]}/{v[1]}" for s, v in sumfar.items()))
    allp, allt = np.array(allp), np.array(allt)
    print(f"  far-only ridge, held out: within a factor of 1.5 {100 * ((allp / allt < 1.5) & (allt / allp < 1.5)).mean():.0f}%, of 2 {100 * ((allp / allt < 2) & (allt / allp < 2)).mean():.0f}%; log correlation {np.corrcoef(np.log2(allp), np.log2(allt))[0, 1]:.2f}")
    print("  near shots wrongly called far, fresh runs: " + " ".join(f"(held {a}) {b}:{c}/{d}" for a, b, c, d in wrong_near))

    print("\n=== Candidate on all far runs, fresh mode 28: near ridge unless called far; what would the radar show?")
    Xf = np.array([x[0]["embed"] for m in FAR for x in data[m][1]], dtype=np.float64)
    Tf = np.array([x[3] for m in FAR for x in data[m][1]])
    g = logistic(np.concatenate([DX, Xf]), np.concatenate([2 ** Dy > CUT, Tf > CUT]))
    for run, rows in fresh_runs.items():
        t = np.array([r[1] for r in rows])
        p = np.array([float(near0(r[0])) for r in rows])
        called = np.array([g(r[0]) > 0 for r in rows])
        for lo, hi in ((0, 42), (42, 65), (65, 88), (88, 130)):
            m = (t >= lo) & (t < hi)
            if m.sum():
                print(f"    {run[-6:]} {lo}-{hi} m: {called[m].sum()}/{m.sum()} called far")
        k = ~called
        err = np.abs(p[k] - t[k])
        print(f"  {run[-6:]}: {k.sum()} shown a distance: median error {np.median(err):.1f} m, within 25% {100 * (err <= 0.25 * t[k]).mean():.1f}%, corr {np.corrcoef(np.log2(p[k]), np.log2(t[k]))[0, 1]:.2f}; {called.sum()} shown 100+ m")
    print("\n=== Candidate namer: labels as tag _off")
    far_rows = {n: gunstorm_rows(n, "_far") for n in FAR}
    A, B = fit(base + [r for m in FAR for r in far_rows[m]])
    word = lambda x: CLASSES[int(np.argmax(A @ np.asarray(x, dtype=np.float64) + B))]
    keys = ["221544", "223129", "224711", "144529", "145757", "152249", "151013", "121745", "135013", "085810", "110232", "133615", "160723", "162444", "215004", "215827", "220700"]
    for k in keys:
        run = U + "20261104_" + k
        tag = next((t for t in ("_base", "_vote", "_app", "_emb") if os.path.exists(f"{run}/replay{t}_s0.jsonl") and "embed" in json.loads(open(f"{run}/replay{t}_s0.jsonl", encoding="utf-8").readline())), None)
        assert tag, k
        with open(f"{run}/replay_off_s0.jsonl", "w", encoding="utf-8") as out:
            for line in open(f"{run}/replay{tag}_s0.jsonl", encoding="utf-8"):
                e = json.loads(line)
                w_ = word(e["embed"])
                e["label"] = "gunshot" if w_ == "gunfire" else (w_.replace("_", " ") if w_ != "other" else (e["clap"] if e["clap"] not in ("gunshot", "machine gun fire") else "rain"))
                out.write(json.dumps({k2: v for k2, v in e.items() if k2 not in ("scores", "embed")}) + "\n")
        if not os.path.exists(f"{run}/replay_base_s0.jsonl") and k in ("215004", "215827", "220700"):
            print("no _base on", k)
    print("written")
