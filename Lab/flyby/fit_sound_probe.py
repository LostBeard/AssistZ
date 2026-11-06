"""
Fit the sound probe AssistZ ships, as JSON for SpawnDev.ILGPU.ML's ClapProbe.FromJson: other, gunfire, gas_strike, heli_crash, sleigh_crash, thunder, infected_attack. The multinomial form of ClapProbe.FitLogistic. Labels come from the game: shot logs for gunfire, world.csv for crashes, and for a gas
strike the Artillery_Close sample match, because a strike sounds minutes after its trigger. The samples are read locally from the player's install and are not committed or shipped; only the weights are.
Usage: python fit_sound_probe.py <samples dir> <out.json> <checkpoint>
"""
import json, multiprocessing as mp, os, sys
import numpy as np
from world_classes import GATE, WORLD, STORM, ATTACK, SURVEY, GUNSTORM, FAR, NEWGUNS, OWN, U, CLASSES, gun_rows, world_rows, storm_rows, attack_rows, survey_rows, gunstorm_rows, init, strike, triggers

from own_cv import own_rows

if __name__ == "__main__":
    samples, out, checkpoint = sys.argv[1], sys.argv[2], sys.argv[3]
    with mp.Pool(4, initializer=init, initargs=(os.path.join(samples, "world"),)) as pool:
        located = pool.map(strike, [(n, when, dist) for n in WORLD for when, ev, dist in triggers(n) if ev == "gas"])
    strikes = {(n, t, d): b for n, t, d, b in located}
    d = [r for n in GATE for r in gun_rows(n, "_emb")] + [r for n in WORLD for r in world_rows(n, strikes)] + [r for n in STORM for r in storm_rows(n)] + [r for n in ATTACK for r in attack_rows(n)] + [r for n in SURVEY for r in survey_rows(n)] + [r for n in GUNSTORM for r in gunstorm_rows(n)] + [r for n in FAR for r in gunstorm_rows(n, "_far")] + [r for n in NEWGUNS for r in gunstorm_rows(n, "_w")] + [r for n in OWN for r in own_rows(U + n)]
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
    probe = dict(Classes=CLASSES, Dim=dim, Weights=[[float(np.float32(v)) for v in row] for row in A], Bias=[float(np.float32(v)) for v in B], Checkpoint=checkpoint,
                 Fit=dict(Steps=800, LearningRate=0.5, L2=1.0, ClassBalanced=True, ExamplesPerClass=[int(c) for c in count]))
    with open(out, "w", encoding="ascii", newline="\n") as f:
        json.dump(probe, f, separators=(",", ":"))
    got = np.argmax(X @ A.T + B, axis=1)
    print(f"{n} examples {dict(zip(CLASSES, count.tolist()))}; training accuracy by class " + str({c: round(float((got[y == i] == i).mean()), 3) for i, c in enumerate(CLASSES)}) + f"; {os.path.getsize(out)} bytes")
