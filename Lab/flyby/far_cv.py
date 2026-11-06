"""
Far gunfire, each far run held out in turn: the namer (shipped training rows plus the other far runs) and the distance ridge (shipped rows plus the other far runs). Then a candidate on every far run, its labels written beside each older holdout as replay_off_s0.jsonl so phase_gates7.py can score it. Raw word only: no burst vote, so a screen, not the gate.
Usage (from Lab/flyby): python far_cv.py <samples dir> <cache.pkl>
"""
import glob, json, multiprocessing as mp, os, pickle, sys
import numpy as np
from world_classes import GATE, WORLD, STORM, ATTACK, SURVEY, GUNSTORM, CLASSES, U, gun_rows, world_rows, storm_rows, attack_rows, survey_rows, gunstorm_rows, init, strike, triggers
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

FAR = ["20261105_055044", "20261105_060740", "20261105_062337", "20261105_063920"]
NAMES = {"055044": "clear", "060740": "clear", "062337": "heavy rain", "063920": "storm"}
STEPS = (150, 200, 300, 400, 600, 800)
near_step = lambda d: min(STEPS, key=lambda s: abs(s - d))


def fit(d):
    X = np.stack([r["x"] for r in d]).astype(np.float64)
    y = np.array([CLASSES.index(r["y"]) for r in d])
    K, n, dim = len(CLASSES), X.shape[0], X.shape[1]
    count = np.bincount(y, minlength=K)
    w = 1.0 / (K * count[y])
    A, B = np.zeros((K, dim)), np.zeros(K)
    Y = np.eye(K)[y]
    for _ in range(800):
        z = X @ A.T + B
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        r = (p - Y) * w[:, None]
        A -= 0.5 * (r.T @ X + 1.0 / n * A)
        B -= 0.5 * r.sum(axis=0)
    return A, B


def far_shots(name):
    """(event, range step, weapon, true distance) for each shot with an event, and the events with no shot within 1.5 s."""
    reports, _ = calibrated_reports(U + name)
    ev = [json.loads(x) for x in open(f"{U}{name}/replay_far_s0.jsonl", encoding="utf-8")]
    shots, none = [], []
    for r in reports:
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        if near:
            shots.append((min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0])), near_step(r[3]), r[2], r[3]))
    for e in ev:
        if all(abs(e["sample"] / 48000.0 - r[0]) > 1.5 for r in reports):
            none.append(e)
    return len(reports), shots, none


def distance_rows():
    """Every labelled gunfire recording in the shipped distance fit, plus nothing far."""
    xs, ys = [], []
    for path in sorted(glob.glob(U + "*/shots_ai.csv")):
        run = os.path.dirname(path).replace("\\", "/")
        tag = next((t for t in ("_dist", "_dir", "_rev", "_base", "_vote", "_emb", "_app") if os.path.exists(f"{run}/replay{t}_s0.jsonl")), None)
        if not tag:
            continue
        first = open(f"{run}/replay{tag}_s0.jsonl", encoding="utf-8").readline()
        if not first or "embed" not in json.loads(first):
            continue
        reports, _ = calibrated_reports(run)
        ev = [json.loads(x) for x in open(f"{run}/replay{tag}_s0.jsonl", encoding="utf-8")]
        for r in reports:
            near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
            if near and r[3] > 0:
                e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0]))
                xs.append(e["embed"])
                ys.append(np.log2(r[3]))
    return np.array(xs, dtype=np.float64), np.array(ys)


def ridge(X, y):
    mx, sx = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = (X - mx) / sx
    w = np.linalg.solve(Z.T @ Z + 100.0 * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda q: 2 ** (((np.asarray(q, dtype=np.float64) - mx) / sx) @ w + y.mean())


if __name__ == "__main__":
    samples, cache = sys.argv[1], sys.argv[2]
    if os.path.exists(cache):
        base, DX, Dy = pickle.load(open(cache, "rb"))
    else:
        with mp.Pool(4, initializer=init, initargs=(os.path.join(samples, "world"),)) as pool:
            located = pool.map(strike, [(n, when, dist) for n in WORLD for when, ev, dist in triggers(n) if ev == "gas"])
        strikes = {(n, t, d): b for n, t, d, b in located}
        base = [r for n in GATE for r in gun_rows(n, "_emb")] + [r for n in WORLD for r in world_rows(n, strikes)] + [r for n in STORM for r in storm_rows(n)] + [r for n in ATTACK for r in attack_rows(n)] + [r for n in SURVEY for r in survey_rows(n)] + [r for n in GUNSTORM for r in gunstorm_rows(n)]
        DX, Dy = distance_rows()
        pickle.dump((base, DX, Dy), open(cache, "wb"))
    print(f"namer base {len(base)} rows; distance base {len(Dy)} rows")
    far_rows = {n: gunstorm_rows(n, "_far") for n in FAR}
    data = {n: far_shots(n) for n in FAR}
    A0, B0 = fit(base)
    word = lambda A, B, x: CLASSES[int(np.argmax(A @ np.asarray(x, dtype=np.float64) + B))]
    d0 = ridge(DX, Dy)
    print("\n=== Namer, each far run held out. Named gunfire / events at a shot, by range; shipped rows only in ()")
    tot = {s: [0, 0, 0] for s in STEPS}
    words = {}
    for n in FAR:
        A, B = fit(base + [r for m in FAR if m != n for r in far_rows[m]])
        rep, shots, none = data[n]
        line = []
        for s in STEPS:
            g = [x for x in shots if x[1] == s]
            new = sum(word(A, B, x[0]["embed"]) == "gunfire" for x in g)
            old = sum(word(A0, B0, x[0]["embed"]) == "gunfire" for x in g)
            line.append(f"{s}:{new}/{len(g)}({old})")
            tot[s][0] += new; tot[s][1] += len(g); tot[s][2] += old
        false_new = sum(word(A, B, e["embed"]) == "gunfire" for e in none)
        false_old = sum(word(A0, B0, e["embed"]) == "gunfire" for e in none)
        print(f"  {n[-6:]} {NAMES[n[-6:]]:<10} {' '.join(line)}  | no shot within 1.5 s: called gunfire {false_new}/{len(none)} ({false_old})")
        for x in shots:
            w_ = word(A, B, x[0]["embed"])
            words.setdefault(x[2], {}).setdefault(w_, 0)
            words[x[2]][w_] += 1
    print("  all: " + " ".join(f"{s}:{v[0]}/{v[1]}({v[2]})" for s, v in tot.items()))
    print("  by weapon: " + str(words))
    print("\n=== Distance, each far run held out. Median estimate by true range; shipped rows only in ()")
    allp, allt = [], []
    for n in FAR:
        X = np.concatenate([DX] + [np.array([x[0]["embed"] for x in data[m][1]], dtype=np.float64) for m in FAR if m != n])
        y = np.concatenate([Dy] + [np.log2([x[3] for x in data[m][1]]) for m in FAR if m != n])
        f = ridge(X, y)
        line = []
        for s in STEPS:
            g = [x for x in data[n][1] if x[1] == s]
            if g:
                p = [float(f(x[0]["embed"])) for x in g]
                q = [float(d0(x[0]["embed"])) for x in g]
                line.append(f"{s}:{np.median(p):.0f}({np.median(q):.0f})")
                allp += p; allt += [x[3] for x in g]
        print(f"  {n[-6:]} {NAMES[n[-6:]]:<10} {' '.join(line)}")
    allp, allt = np.array(allp), np.array(allt)
    print(f"  all far held out: within a factor of 1.5 {100 * ((allp / allt < 1.5) & (allt / allp < 1.5)).mean():.0f}%, of 2 {100 * ((allp / allt < 2) & (allt / allp < 2)).mean():.0f}%; log correlation {np.corrcoef(np.log2(allp), np.log2(allt))[0, 1]:.2f}; n {len(allp)}")
    print("\n=== Near gunfire stays right? Fresh mode 28 runs, candidate fit on everything far (shipped limits 10 m, 75%, 0.7)")
    X = np.concatenate([DX] + [np.array([x[0]["embed"] for x in data[m][1]], dtype=np.float64) for m in FAR])
    y = np.concatenate([Dy] + [np.log2([x[3] for x in data[m][1]]) for m in FAR])
    f = ridge(X, y)
    for run in ("20261105_051052", "20261105_052232"):
        reports, _ = calibrated_reports(U + run)
        ev = [json.loads(x) for x in open(f"{U}{run}/replay_ship_s0.jsonl", encoding="utf-8")]
        t, p, q = [], [], []
        for r in reports:
            if r[3] > 150:
                continue
            near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
            if near:
                e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0]))
                t.append(r[3]); p.append(float(f(e["embed"]))); q.append(float(d0(e["embed"])))
        t, p, q = np.array(t), np.array(p), np.array(q)
        print(f"  {run[-6:]}: candidate median error {np.median(np.abs(p - t)):.1f} m, within 25% {100 * (np.abs(p - t) <= 0.25 * t).mean():.1f}%, corr {np.corrcoef(np.log2(p), np.log2(t))[0, 1]:.2f} | shipped rows {np.median(np.abs(q - t)):.1f} m, {100 * (np.abs(q - t) <= 0.25 * t).mean():.1f}%, {np.corrcoef(np.log2(q), np.log2(t))[0, 1]:.2f}")
    print("\n=== Candidate namer on everything far: labels written as tag _off for phase_gates7.py")
    A, B = fit(base + [r for m in FAR for r in far_rows[m]])
    keys = ["221544", "223129", "224711", "144529", "145757", "152249", "151013", "121745", "135013", "085810", "110232", "133615", "160723", "162444", "215004", "215827", "220700"]
    for k in keys:
        run = U + "20261104_" + k
        src = f"{run}/replay_base_s0.jsonl"
        n = 0
        with open(f"{run}/replay_off_s0.jsonl", "w", encoding="utf-8") as out:
            for line in open(src, encoding="utf-8"):
                e = json.loads(line)
                w_ = word(A, B, e["embed"])
                e["label"] = "gunshot" if w_ == "gunfire" else (e["clap"] if w_ == "other" and e["clap"] not in ("gunshot", "machine gun fire") else ("rain" if w_ == "other" else w_.replace("_", " ")))
                out.write(json.dumps({k2: v for k2, v in e.items() if k2 not in ("scores", "embed")}) + "\n")
                n += 1
    print("written")
