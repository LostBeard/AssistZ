"""
Does the app decide what the fitted five-class JSON says, event by event? And its score on recordings it never trained on.
Usage: python check_app_classes.py <probe.json> <tag> <run> [...]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

probe = json.load(open(sys.argv[1]))
A, B, C = np.array(probe["Weights"], dtype=np.float64), np.array(probe["Bias"], dtype=np.float64), probe["Classes"]
TAG, RUNS = sys.argv[2], sys.argv[3:]
WORD = {"gas_strike": "gas strike", "heli_crash": "heli crash", "sleigh_crash": "sleigh crash", "thunder": "thunder", "infected_attack": "infected attack"}
GUN = ("gunshot", "automatic fire")
total = agree = 0
tg = cg = to = co = 0
worlds = {}
for run in RUNS:
    reports, _ = calibrated_reports(run)
    events = [json.loads(x) for x in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8")]
    g = gh = o = oh = 0
    words = {}
    for e in events:
        c = C[int(np.argmax(A @ np.array(e["embed"], dtype=np.float64) + B))]
        app = e.get("raw", e["label"])
        app = "gunfire" if app in GUN else app
        expected = WORD.get(c, c)
        total += 1
        agree += (app == expected) if c != "other" else (app not in GUN and app not in WORD.values())
        t = e["sample"] / 48000.0
        if any(-0.05 <= t - r[0] <= 0.15 for r in reports):
            g += 1; gh += e["label"] in GUN
        elif all(abs(t - r[0]) > 1.5 for r in reports):
            o += 1; oh += e["label"] in GUN
            words[e["label"]] = words.get(e["label"], 0) + 1
            if e["label"] in WORD.values():
                worlds.setdefault((os.path.basename(run.rstrip("/"))[-6:], e["label"]), []).append(round(t, 1))
    print(f"  {os.path.basename(run.rstrip('/'))[-15:]}: {len(events)} events; gunfire {gh}/{g}; no shot called gunfire {oh}/{o}; labels {dict(sorted(words.items(), key=lambda x: -x[1])[:6])}")
    tg += g; cg += gh; to += o; co += oh
print(f"  app decision equals the JSON's: {agree}/{total}")
print(f"  TOTAL: gunfire {cg}/{tg}; no shot called gunfire {co}/{to}")
print("  named world events (recording, word: seconds):")
for k, v in sorted(worlds.items()):
    print(f"    {k[0]} {k[1]:<13} {len(v):3d}: {v[:12]}")
