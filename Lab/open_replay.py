"""
The candidate probe on the lab's real game mixes, from embeddings already in each run's replay: writes replay<tag>_s0.jsonl with the label the app's rule would give, so the old gate scripts read it unchanged. Fits on every game sound and every training recording. Usage: open_replay.py <tag> [L2]. Tag _open.
"""
import glob, json, os, sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "D:/users/tj/Projects/AssistZ/AssistZ/Lab")
from open_data import BANK, CLASSES, bank
TAG = sys.argv[1]
L2 = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
LABW = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
OUT = os.path.join(BANK, f"open_all{TAG}.npz")
RUNS = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_"
K, DIM = len(CLASSES), 512
BAR = np.full(K, 0.3)
for c in ("gas_strike", "heli_crash"):
    BAR[CLASSES.index(c)] = 0.5
OWN = [CLASSES.index("own_move"), CLASSES.index("own_shot")]
WORD = {"gunfire": "gunshot", "gas_strike": "gas strike", "heli_crash": "heli crash", "infected_attack": "infected attack", "own_move": "own move", "own_shot": "own shot"}

if not os.path.exists(OUT):
    rows, emb, y, sp, names, subs = bank()
    lab = np.load(os.path.join(BANK, "open_lab.npz"))
    lx, ly = lab["X"].astype(np.float64), np.array([CLASSES.index(c) for c in lab["y"]])
    m = y >= 0
    X = np.concatenate([emb[m].reshape(-1, DIM).astype(np.float64), lx])
    Y = np.concatenate([np.repeat(y[m], emb.shape[1]), ly])
    src = np.concatenate([np.ones(2 * m.sum()), LABW * np.ones(len(ly))])
    count = np.array([src[Y == i].sum() for i in range(K)])
    w = src / (K * np.maximum(count[Y], 1))
    n = len(Y)
    O = np.zeros((n, K))
    O[np.arange(n), Y] = 1

    def loss(v):
        A, B = v[:K * DIM].reshape(K, DIM), v[K * DIM:]
        z = X @ A.T + B
        mx = z.max(axis=1, keepdims=True)
        l = float((w * (mx[:, 0] + np.log(np.exp(z - mx).sum(axis=1)) - z[np.arange(n), Y])).sum()) + 0.5 * L2 * float((A * A).sum())
        p = np.exp(z - mx)
        p /= p.sum(axis=1, keepdims=True)
        r = (p - O) * w[:, None]
        return l, np.concatenate([(r.T @ X + L2 * A).ravel(), r.sum(axis=0)])

    res = minimize(loss, np.zeros(K * DIM + K), jac=True, method="L-BFGS-B", options=dict(maxiter=400, maxcor=20))
    np.savez(OUT, A=res.x[:K * DIM].reshape(K, DIM), B=res.x[K * DIM:], classes=np.array(CLASSES))
    print(f"fitted on {n} examples, {res.nit} steps; per class", {CLASSES[i]: int((Y == i).sum()) for i in range(K)}, flush=True)
d = np.load(OUT)
A, B = d["A"], d["B"]


def label(emb, az):
    z = A @ emb + B
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    if abs(az) >= 10 or p[OWN[0]] < 0.55:
        p[OWN[0]] = -1
    if abs(az) >= 10 or p[OWN[1]] < 0.55:
        p[OWN[1]] = -1
    top = int(p.argmax())
    if p[top] < BAR[top]:
        return "unnamed", float(p[top])
    return WORD.get(CLASSES[top], CLASSES[top]), float(p[top])


wrote = skipped = 0
for run in sorted(glob.glob(RUNS + "*")):
    src = next((f"{run}/replay{t}_s0.jsonl" for t in ("_g", "_r7", "_f3", "_f2", "_f1", "_own", "_app", "_vote", "_emb", "_w", "_far", "_st", "_base") if os.path.exists(f"{run}/replay{t}_s0.jsonl")), None)
    if not src:
        continue
    ev = [json.loads(x) for x in open(src, encoding="utf-8")]
    if not ev or "embed" not in ev[0]:
        skipped += 1
        continue
    with open(f"{run}/replay{TAG}_s0.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for e in ev:
            word, conf = label(np.array(e["embed"], dtype=np.float64), e.get("az", e.get("azimuth", 99)))
            e2 = {k: v for k, v in e.items() if k != "embed"}
            e2["label"], e2["p"] = word, conf
            f.write(json.dumps(e2) + "\n")
    wrote += 1
print(f"wrote {wrote} replays, {skipped} had no embeddings")
