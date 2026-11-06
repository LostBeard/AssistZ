"""
Would a vote across a burst help? Each world run held out in turn, as world_classes.py: per-event labels, then with world-event words replaced by their burst's majority. A burst = events with gaps under <gap> s. Gunfire is never changed. Also: where do capture-start events fall?
Usage: python vote_sim.py <samples dir>
"""
import glob, json, multiprocessing as mp, os, sys
import numpy as np
from world_classes import U, GATE, WORLD, HOLD, CLASSES, gun_rows, world_rows, fit, init, strike, triggers

WORLDS = ("gas_strike", "heli_crash", "sleigh_crash")


def vote(seq, gap):
    """seq: (t, word). Returns words after the vote."""
    out = [w for _, w in seq]
    i = 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1][0] - seq[j][0] < gap:
            j += 1
        count = {}
        for k in range(i, j + 1):
            if out[k] in WORLDS:
                count[out[k]] = count.get(out[k], 0) + 1
        if count:
            top = max(count, key=count.get)
            for k in range(i, j + 1):
                if out[k] in WORLDS:
                    out[k] = top
        i = j + 1
    return out


if __name__ == "__main__":
    samples = sys.argv[1]
    with mp.Pool(4, initializer=init, initargs=(os.path.join(samples, "world"),)) as pool:
        located = pool.map(strike, [(n, when, dist) for n in WORLD for when, ev, dist in triggers(n) if ev == "gas"])
    strikes = {(n, t, d): b for n, t, d, b in located}
    world = {n: world_rows(n, strikes) for n in WORLD}
    gate = [r for n in GATE for r in gun_rows(n, "_emb")]
    results = {}
    for n in WORLD:
        f = fit(gate + [r for m in WORLD if m != n for r in world[m]])
        rows = sorted(world[n], key=lambda r: r["t"])
        results[n] = [(r["t"], r["y"], f(r["x"]), r["event"]) for r in rows]
    for gap in (None, 0.5, 1.0, 1.5, 2.5, 4.0):
        conf = {a: {b: 0 for b in CLASSES} for a in CLASSES}
        mixed = bursts = 0
        for n, rows in results.items():
            words = [r[2] for r in rows] if gap is None else vote([(r[0], r[2]) for r in rows], gap)
            for r, w in zip(rows, words):
                conf[r[1]][w] += 1
            # A burst showing more than one world word: what the viewer sees flicker.
            g = gap or 1.0
            i = 0
            while i < len(rows):
                j = i
                while j + 1 < len(rows) and rows[j + 1][0] - rows[j][0] < g:
                    j += 1
                seen = {words[k] for k in range(i, j + 1) if words[k] in WORLDS}
                bursts += bool(seen)
                mixed += len(seen) > 1
                i = j + 1
        name = "no vote" if gap is None else f"gap {gap} s"
        print(f"{name:<10} gas {conf['gas_strike']['gas_strike']}/{sum(conf['gas_strike'].values())}, heli {conf['heli_crash']['heli_crash']}/{sum(conf['heli_crash'].values())}, sleigh {conf['sleigh_crash']['sleigh_crash']}/{sum(conf['sleigh_crash'].values())}; other called a world event {sum(conf['other'][c] for c in WORLDS)}/{sum(conf['other'].values())}; bursts with mixed words {mixed}/{bursts}")
    print("\nDetector events in the first 5 s of each replay (t, word), and the first event after:")
    late = []
    for path in sorted(glob.glob(U + "2026110[34]_*/replay_cls_s0.jsonl") + glob.glob(U + "2026110[34]_*/replay_emb_s0.jsonl")):
        name = path.replace("\\", "/").split("/")[-2][-15:]
        ev = [json.loads(x) for x in open(path, encoding="utf-8")]
        first = [(round(e["sample"] / 48000, 2), e["clap"][:8]) for e in ev if e["sample"] / 48000 < 5]
        next_t = next((round(e["sample"] / 48000, 1) for e in ev if e["sample"] / 48000 >= 5), None)
        if "cls" in path or name not in {p for p in late}:
            late.append(name)
            print(f"  {name}: {first}; next {next_t}")
