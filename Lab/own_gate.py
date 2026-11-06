"""
The registered own sounds checks, from the app's own labels. own <tag> <run>: fresh mode 32. others <tag> <run>: another character's footsteps, mode 10. agro <tag> <run>: fresh mode 30.
"""
import collections, csv, datetime, json, os, sys
import numpy as np
kind, tag = sys.argv[1], sys.argv[2]
every = True
for run in [p.replace("\\", "/").rstrip("/") for p in sys.argv[3:]]:
    ev = [json.loads(x) for x in open(f"{run}/replay{tag}_s0.jsonl", encoding="utf-8")]
    name = os.path.basename(run)
    m = json.load(open(run + "/audio.json"))
    st = datetime.datetime.fromisoformat(m["start_utc"])
    s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
    env = list(csv.DictReader(open(run + "/env.csv")))
    et = np.array([int(e["game_ms"]) / 1000 + off - s for e in env])
    worst = np.array([int(e.get("infected_worst", -1) or -1) for e in env]) if "infected_worst" in env[0] else None
    print(f"{name}: {len(ev)} events; {dict(collections.Counter(e['label'] for e in ev).most_common(8))}")
    if kind == "own":
        drive = json.load(open(run + "/own_drive.json"))
        stamp = st.timestamp()
        ph = [(p["name"], datetime.datetime.fromisoformat(p["start_utc"]).timestamp() - stamp, datetime.datetime.fromisoformat(p["end_utc"]).timestamp() - stamp) for p in drive["phases"]]
        shots = np.array([int(r["game_ms"]) / 1000 + off - s for r in csv.DictReader(open(run + "/aishots_server.csv"))]) if os.path.exists(run + "/aishots_server.csv") else np.array([])
        t = np.array([e["sample"] / 48000.0 for e in ev])
        if len(shots):
            shots = shots + max(np.arange(-3, 3, 0.02), key=lambda d: sum(np.any((t - (a + d) >= 0) & (t - (a + d) <= 0.2)) for a in shots))
        move, fire = [], []
        for e, x in zip(ev, t):
            n = next((a for a, lo, hi in ph if lo <= x < hi), None)
            i = int(np.argmin(np.abs(et - x)))
            if worst is not None and worst[i] > 0:
                continue
            if n == "fire":
                if len(shots) and np.any((x - shots >= -0.05) & (x - shots <= 0.4)):
                    fire.append(e)
            elif n and not n.startswith("still"):
                move.append(e)
        bad = sum(e["label"] in ("gunshot", "automatic fire", "infected attack") for e in move)
        own = sum(e["label"] == "own move" for e in move)
        own_s = sum(e["label"] == "own shot" for e in fire)
        checks = [
            (bad / max(len(move), 1) <= 0.05, f"movement named gunfire or infected {bad}/{len(move)} ({100 * bad / max(len(move), 1):.1f}%, at most 5)"),
            (own / max(len(move), 1) >= 0.75, f"movement named own {own}/{len(move)} ({100 * own / max(len(move), 1):.1f}%, at least 75); {dict(collections.Counter(e['label'] for e in move).most_common(5))}"),
            (own_s / max(len(fire), 1) >= 0.60, f"own shots named own {own_s}/{len(fire)} ({100 * own_s / max(len(fire), 1):.1f}%, at least 60); {dict(collections.Counter(e['label'] for e in fire).most_common(4))}"),
        ]
    elif kind == "others":
        own = sum(e["label"].startswith("own") for e in ev)
        checks = [(own / max(len(ev), 1) <= 0.10, f"named own {own}/{len(ev)} ({100 * own / max(len(ev), 1):.1f}%, at most 10)")]
    else:
        state = worst
        agit, calm = [], []
        for e in ev:
            i = int(np.argmin(np.abs(et - e["sample"] / 48000.0)))
            (agit if state[i] in (1, 2, 3, 4) else calm if state[i] == 0 else []).append(e)
        a = sum(e["label"] == "infected attack" for e in agit)
        c = sum(e["label"] == "infected attack" for e in calm)
        checks = [(a / max(len(agit), 1) >= 0.90, f"agitated called agroed {a}/{len(agit)} ({100 * a / max(len(agit), 1):.1f}%, at least 90)"), (c / max(len(calm), 1) <= 0.15, f"calm called agroed {c}/{len(calm)} ({100 * c / max(len(calm), 1):.1f}%, at most 15)")]
    for ok, text in checks:
        print(f"  {'PASS' if ok else 'FAIL'} {text}")
        every = every and ok
print("ALL PASS" if every else "SOME FAILED")
