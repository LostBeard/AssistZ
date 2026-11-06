"""
Export the labelled clips for an embedding probe: one JSON line per clip, with the truth taken from the game's logs. Clip = 48000 samples from `clip`, mono 0.5 (L + R),
from <run>/audio.npy. Events 50 ms before to 150 ms after a report are gunfire; events over 1.5 s from any report are "other"; those between are left out. In a
no-gunfire recording (lab mode 10), events with the AI walking within 35 m are "footsteps". `split` is train, or test for recordings kept out of training.
Usage: python export_label_set.py <out.jsonl> <tag> train <run> [...] test <run> [...]
"""
import collections, csv, datetime, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_clock import calibrated_reports

OUT, TAG = sys.argv[1], sys.argv[2]
args = sys.argv[3:]
split = None
runs = []
for a in args:
    if a in ("train", "test"):
        split = a
    else:
        runs.append((a, split))
AUTOMATIC = ("AugShort", "AKM", "M4A1")
count = collections.Counter()
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    for run, sp in runs:
        reports, _ = calibrated_reports(run)
        walk = None
        if not reports and os.path.exists(run + "/aipose.csv"):
            meta = json.load(open(run + "/audio.json"))
            st = datetime.datetime.fromisoformat(meta["start_utc"])
            start = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
            pose = list(csv.DictReader(open(run + "/pose.csv")))
            offset = np.median([sum(int(v) * k for v, k in zip(r["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(r["game_ms"]) / 1000 for r in pose])
            walk = [(int(x["game_ms"]) / 1000 + offset - start + float(x["dist"]) / 330.0, float(x["speed"]), float(x["dist"])) for x in csv.DictReader(open(run + "/aipose.csv"))]
        for line in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8"):
            e = json.loads(line)
            t = e["sample"] / 48000.0
            near = [r for r in reports if -0.05 <= t - r[0] <= 0.15]
            row = dict(run=os.path.basename(run.rstrip("/\\")), split=sp, clip=e["clip"], t=round(t, 3), clap=e["clap"], p=e["p"])
            if near:
                row.update(truth="gunfire", weapon=near[0][2], fire="automatic" if near[0][2] in AUTOMATIC else "single", distance=round(near[0][3], 1))
            elif walk is not None:
                best = min(walk, key=lambda x: abs(x[0] - t))
                row["truth"] = "footsteps" if abs(best[0] - t) < 1.0 and best[1] > 0.5 and best[2] < 35 else "other"
            elif all(abs(t - r[0]) > 1.5 for r in reports):
                row["truth"] = "other"
            else:
                continue
            f.write(json.dumps(row) + "\n")
            count[(sp, row["truth"], row.get("weapon", ""))] += 1
print("wrote", OUT)
for k in sorted(count):
    print(" ", k, count[k])
