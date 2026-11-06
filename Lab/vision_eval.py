"""
Where am I, from one frame? Index: grid 26. Query: grid 27, offset by half a cell, so the nearest possible index point is 707 m away and no query place was indexed. Scored against shuffled positions. Targets registered in docs/PLAN.md before any of these numbers were seen.
Usage: python vision_eval.py <frames dir> <model> [...]   (expects <frames dir>/grid26_<model>.npz, grid27_<model>.npz)
"""
import csv, os, sys
import numpy as np

ROOT, MODELS = sys.argv[1], sys.argv[2:]
TILES = ("tile0", "tile1", "tile2")


def load(grid, model):
    z = np.load(os.path.join(ROOT, f"{grid}_{model}.npz"))
    rows = {r["frame"]: r for r in csv.DictReader(open(os.path.join(ROOT, grid, "manifest.csv")))}
    names = z["n_center"]
    keep = [i for i, n in enumerate(names) if n in rows]
    meta = [rows[names[i]] for i in keep]
    pos = np.array([[float(m["x"]), float(m["z"])] for m in meta])
    head = np.array([int(float(m["heading"])) for m in meta])
    vec = {k: z["v_" + k][keep].astype(np.float32) for k in ("center",) + TILES if "v_" + k in z}
    return pos, head, vec


def score(sim, ip, ih, qp, qh, title):
    best = np.argmax(sim, axis=1)
    err = np.linalg.norm(ip[best] - qp, axis=1)
    top5 = np.argsort(-sim, axis=1)[:, :5]
    near5 = np.array([np.linalg.norm(ip[t] - p, axis=1).min() for t, p in zip(top5, qp)])
    rng = np.random.default_rng(7)
    chance = [np.median(np.linalg.norm(ip[rng.permutation(len(ip))[best]] - qp, axis=1)) for _ in range(20)]
    floor = np.array([np.linalg.norm(ip - p, axis=1).min() for p in qp])
    print(f"  {title:<26} median {np.median(err):6.0f} m (shuffled {np.mean(chance):6.0f}, nearest possible {np.median(floor):4.0f}); within 1 km {100 * (err <= 1000).mean():4.1f}%, 1.5 km {100 * (err <= 1500).mean():4.1f}%, 3 km {100 * (err <= 3000).mean():4.1f}%; best of top 5 within 1 km {100 * (near5 <= 1000).mean():4.1f}%; heading {100 * (ih[best] == qh).mean():4.1f}% (chance 25)")
    return np.median(err), (ih[best] == qh).mean(), err


for model in MODELS:
    ip, ih, iv = load("grid26", model)
    qp, qh, qv = load("grid27", model)
    print(f"{model}: index {len(ip)} views, query {len(qp)} views")
    score(qv["center"] @ iv["center"].T, ip, ih, qp, qh, "whole frame, centre crop")
    if all(t in qv for t in TILES):
        # Each query tile finds its best match among all index tiles; a view scores the mean of the three.
        per = []
        for qt in TILES:
            s = np.stack([qv[qt] @ iv[it].T for it in TILES])
            per.append(s.max(axis=0))
        score(np.mean(per, axis=0), ip, ih, qp, qh, "three tiles, no HUD")
        m = np.mean([qv[t] @ iv[t].T for t in TILES], axis=0)
        score(m, ip, ih, qp, qh, "three tiles, same position")
    # Where it does well and badly: by the point's distance to the sea edge is not known; by how often the same index area wins.
    best = np.argmax(qv["center"] @ iv["center"].T, axis=1)
    counts = np.bincount(best, minlength=len(ip))
    print(f"  most-picked index view chosen {counts.max()} times of {len(qp)}; distinct index views chosen {(counts > 0).sum()} of {len(ip)}")
