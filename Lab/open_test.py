"""
The written test of the open-set probe, run once: held-out test kinds of sound, whole guns never trained on, and ESC-50 from outside the game. Bars as PLAN fixed them: 0.3, and 0.5 for gas strike and helicopter crash. Usage: open_test.py
"""
import collections, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from open_data import BANK, CLASSES, GUNLIKE, bank
d = np.load(os.path.join(BANK, "open_probe.npz"))
A, B = d["A"], d["B"]
BAR = np.full(len(CLASSES), 0.3)
for c in ("gas_strike", "heli_crash"):
    BAR[CLASSES.index(c)] = 0.5
OWN = [CLASSES.index("own_move"), CLASSES.index("own_shot")]
GUN, GAS, ATT = CLASSES.index("gunfire"), CLASSES.index("gas_strike"), CLASSES.index("infected_attack")


def call(e):
    """Class said for each clip, or -1 for no name. Away from centre: own classes cannot win."""
    z = e @ A.T + B
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    p[:, OWN] = -1
    top = p.argmax(axis=1)
    conf = p[np.arange(len(p)), top]
    top = top.copy()
    top[conf < BAR[top]] = -1
    return top


rows, emb, y, sp, names, subs = bank()
te = sp == "test"
print("guns in test:", sorted(set(subs[te & (names == "gunshot")].tolist())), "; test clips", int(te.sum()), "; kinds of group", len(set(names[te].tolist())))
results = []
for lv, level in enumerate((-6, -24)):
    c = call(emb[:, lv])
    notgun = te & ~np.isin(names, list(GUNLIKE))
    notart = te & ~np.isin(names, ["artillery", "explosion"])
    notinf = te & ~np.isin(names, ["infected voice", "infected movement"])
    real = te & (names == "gunshot")
    fg, fgas, fatt, rg = 100 * (c[notgun] == GUN).mean(), 100 * (c[notart] == GAS).mean(), 100 * (c[notinf] == ATT).mean(), 100 * (c[real] == GUN).mean()
    print(f"\n=== test kinds, {level} dB")
    print(f"LIMIT false gunfire {fg:.2f}% (at most 1) of {notgun.sum()}: {'PASS' if fg <= 1 else 'FAIL'}")
    print(f"LIMIT false gas strike {fgas:.2f}% (at most 1) of {notart.sum()}: {'PASS' if fgas <= 1 else 'FAIL'}")
    print(f"LIMIT false infected attack {fatt:.2f}% (at most 2) of {notinf.sum()}: {'PASS' if fatt <= 2 else 'FAIL'}")
    need = 85 if level == -6 else 70
    print(f"LIMIT gunshots from guns never trained on {rg:.1f}% (at least {need}) of {real.sum()}: {'PASS' if rg >= need else 'FAIL'}")
    print(f"unnamed: {100 * (c[te] == -1).mean():.1f}% of test clips")
    print(f"  {'group':<20}{'n':>6} {'no name':>8}  top names (%)")
    for g in sorted(set(names[te].tolist()), key=lambda g: -(names[te] == g).sum()):
        k = te & (names == g)
        top = collections.Counter(c[k].tolist())
        none = 100 * top.pop(-1, 0) / k.sum()
        print(f"  {g:<20}{k.sum():>6} {none:>7.0f}%  " + ", ".join(f"{CLASSES[i]} {100 * v / k.sum():.0f}" for i, v in top.most_common(3)))
    print("  gunshots by gun:", {g: f"{100 * (c[real & (subs == g)] == GUN).mean():.0f}% of {(real & (subs == g)).sum()}" for g in sorted(set(subs[real].tolist()))})
E = "L:/AssistZLabWork/esc50/bank"
meta = list(csv.DictReader(open(E + "/manifest.csv", encoding="utf-8")))
esc = np.load(E + "/emb_000.npy")
cats = np.array([m["category"] for m in meta])
for lv, level in enumerate((-6, -24)):
    c = call(esc[:, lv])
    fg, fgas = 100 * (c == GUN).mean(), 100 * (c == GAS).mean()
    print(f"\n=== ESC-50, outside the game, {level} dB, {len(c)} clips")
    print(f"LIMIT false gunfire {fg:.2f}% (at most 2): {'PASS' if fg <= 2 else 'FAIL'}")
    print(f"LIMIT false gas strike {fgas:.2f}% (at most 2): {'PASS' if fgas <= 2 else 'FAIL'}")
    print(f"unnamed {100 * (c == -1).mean():.1f}%; false infected attack {100 * (c == ATT).mean():.2f}%")
    worst = sorted(set(cats.tolist()), key=lambda k: -((c[cats == k] == GUN).mean() + (c[cats == k] == GAS).mean()))[:6]
    print("  most often gunfire or gas strike:", {k: f"{100 * (c[cats == k] == GUN).mean():.0f}/{100 * (c[cats == k] == GAS).mean():.0f}" for k in worst})
    print("  the names given, overall:", {("no name" if i == -1 else CLASSES[i]): f"{100 * v / len(c):.0f}%" for i, v in collections.Counter(c.tolist()).most_common(8)})
