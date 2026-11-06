"""
Replication of round 4's one failed gate, on sites never heard. Frozen candidate 2E5E6A463D2B, no refit. Registered in the handoff before these runs. Survey labels as round 3: leave out 8 s after each move, an infected under 3 m, -2..+12 s around an Artillery_Close match of 0.4 or more. Gate: at least 100 scored events; at most 5% named gas strike, heli crash, thunder or infected attack; 0 gunfire; no site with at least 5 events over 20%.
Usage: python replicate_gate.py <tag> <run dir> [...]
"""
import csv, datetime, json, sys
import numpy as np

TAG, RUNS = sys.argv[1], sys.argv[2:]
FALSE = ("gas strike", "heli crash", "sleigh crash", "thunder", "infected attack")
GUN = ("gunshot", "automatic fire")
scored = []
sites = {}
for run in RUNS:
    m = json.load(open(run + "/audio.json"))
    st = datetime.datetime.fromisoformat(m["start_utc"])
    s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
    world = [(int(e["game_ms"]) / 1000 + off - s, float(e["distance"])) for e in csv.DictReader(open(run + "/world.csv")) if e["event"] == "site"]
    env = [(int(e["game_ms"]) / 1000 + off - s, float(e["infected_nearest"])) for e in csv.DictReader(open(run + "/env.csv"))]
    et, near = np.array([x[0] for x in env]), np.array([x[1] for x in env])
    strikes = json.load(open(run + "/strikes.json"))
    ev = [json.loads(x) for x in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8")]
    kept = 0
    for e in ev:
        t = e["sample"] / 48000.0
        if t < world[0][0] or any(w <= t < w + 8 for w, x in world):
            continue
        d = near[np.argmin(np.abs(et - t))]
        if 0 <= d < 3 or any(at - 2 <= t <= at + 12 for ncc, at in strikes):
            continue
        site = (run[-6:], max(x for w, x in world if w <= t))
        scored.append(e)
        sites.setdefault(site, []).append(e)
        kept += 1
    print(f"{run[-15:]}: {len(ev)} events, {kept} scored, strikes {strikes}")
n = max(len(scored), 1)
false = sum(e["label"] in FALSE for e in scored)
gun = sum(e["label"] in GUN for e in scored)
words = {}
for e in scored:
    words[e["label"]] = words.get(e["label"], 0) + 1
print("words:", dict(sorted(words.items(), key=lambda x: -x[1])))
big = [(sum(e["label"] in FALSE for e in g) / len(g), k, len(g)) for k, g in sites.items() if len(g) >= 5]
worst = max(big, default=(0, ("-", -1), 0))
print("by site:", {f"{k[0]} {k[1]:.0f}": len(g) for k, g in sorted(sites.items())})
print(f"false words at: {[(k[0], round(k[1]), e['label'], round(e['sample'] / 48000, 1)) for k, g in sites.items() for e in g if e['label'] in FALSE][:20]}")
ok = len(scored) >= 100 and false / n <= 0.05 and gun == 0 and worst[0] <= 0.20
print(f"{'PASS' if ok else 'FAIL'}: {len(scored)} scored (at least 100); false words {false} ({100 * false / n:.1f}%, at most 5); gunfire {gun}; worst site with 5+ events {100 * worst[0]:.0f}% ({worst[1]}, {worst[2]} events; at most 20)")
print("ENOUGH" if len(scored) >= 100 else "NEED MORE")
