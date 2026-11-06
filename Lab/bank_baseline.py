"""What does the SHIPPED sound probe call the game's real sounds? The false alarm measure the lab never had. Per group of the game's own folders: share of clips called each class, at both levels, as a real sound away from centre (own classes off) would be. Usage: bank_baseline.py"""
import collections, json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from bank_lib import group, load
rows, emb = load()
probe = json.load(open("D:/users/tj/Projects/AssistZ/AssistZ/AssistZ/wwwroot/models/sound-probe.json"))
C = probe["Classes"]
W, B = np.array(probe["Weights"]), np.array(probe["Bias"])
print("classes", C, "weights", W.shape, "bias", B.shape)
if W.shape[0] != len(C):
    W = W.reshape(len(C), -1)
own = [i for i, c in enumerate(C) if c.startswith("own_")]
groups = [group(r) for r in rows]
names = collections.Counter(g[0] for g in groups)
print({k: v for k, v in names.most_common()})
order = [k for k, _ in names.most_common()]
for lv, level in enumerate((-6, -24)):
    logits = emb[:, lv] @ W.T + B
    # Softmax, as the app does; away from centre, the own classes cannot win.
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    q = p.copy()
    q[:, own] = -1
    top = q.argmax(axis=1)
    print(f"\n=== level {level} dB. Share called each class, %. n = clips")
    print(f"{'group':<20}{'n':>6} " + "".join(f"{c[:9]:>10}" for c in C if not c.startswith("own_")))
    for g in order:
        k = np.array([x[0] == g for x in groups])
        line = f"{g:<20}{k.sum():>6} "
        for i, c in enumerate(C):
            if i in own:
                continue
            line += f"{100 * (top[k] == i).mean():>10.1f}"
        print(line)
    conf = q[np.arange(len(q)), top]
    not_gun = np.array([not x[0].startswith(("gunshot", "gun tail", "explosion", "other weapon")) for x in groups])
    gi = C.index("gunfire")
    print(f"called gunfire among sounds that are not gunshots, explosions or other weapons: {100 * (top[not_gun] == gi).mean():.1f}% of {not_gun.sum()}; at the app's probability, median {np.median(conf[not_gun & (top == gi)]):.2f}")
    real = np.array([x[0] == "gunshot" for x in groups])
    print(f"called gunfire among real gunshots: {100 * (top[real] == gi).mean():.1f}% of {real.sum()}")
