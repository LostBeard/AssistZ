"""
Does the app's probe decide what the fitted JSON says it should? For each replay with embeddings: the shipped weights' sign against the app's label, event by event. Also scores against the shot log.
Usage: python check_app_probe.py <probe.json> <tag> <run> [...]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

probe = json.load(open(sys.argv[1]))
a, b = np.array(probe["Weights"][0], dtype=np.float64), probe["Bias"][0]
TAG, RUNS = sys.argv[2], sys.argv[3:]
GUN = ("gunshot", "automatic fire")
total, agree = 0, 0
tg = cg = to = co = 0
tp = cp = op = 0
for run in RUNS:
    reports, _ = calibrated_reports(run)
    events = [json.loads(x) for x in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8")]
    g = gh = o = oh = pg = po = 0
    words = {}
    for e in events:
        margin = float(np.array(e["embed"], dtype=np.float64) @ a + b)
        app = e["label"] in GUN
        total += 1
        agree += app == (margin > 0)
        t = e["sample"] / 48000.0
        clap_gun = e["clap"] in ("gunshot", "machine gun fire")
        if any(-0.05 <= t - r[0] <= 0.15 for r in reports):
            g += 1; gh += app; pg += clap_gun
        elif all(abs(t - r[0]) > 1.5 for r in reports):
            o += 1; oh += app; po += clap_gun
            words[e["label"]] = words.get(e["label"], 0) + 1
    print(f"  {os.path.basename(run.rstrip('/'))[-15:]}: {len(events)} events; gunfire named gunfire {gh}/{g} (CLAP word {pg}/{g}); no shot called gunfire {oh}/{o} (CLAP word {po}/{o}); labels {dict(sorted(words.items(), key=lambda x: -x[1])[:5])}")
    tg += g; cg += gh; to += o; co += oh; tp += pg; op += po
print(f"  app decision equals the JSON's sign: {agree}/{total}")
print(f"  TOTAL: gunfire {cg}/{tg} (CLAP word {tp}/{tg}); no shot called gunfire {co}/{to} (CLAP word {op}/{to})")
