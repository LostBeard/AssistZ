"""Real gunfire in a storm: what did each probe make of the same clips, and CLAP's own prompts? Same replay embeddings, weights only. Also how many shots reach the detector at all."""
import json, subprocess, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

REPO = r"D:\users\tj\Projects\AssistZ\AssistZ"
U = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_20261104_"
probes = {}
for name, rev in (("before today (5 classes)", "e71a4a7~1"), ("shipped now (6 classes)", "e71a4a7")):
    txt = subprocess.run(["git", "-C", REPO, "show", f"{rev}:AssistZ/wwwroot/models/sound-probe.json"], capture_output=True, text=True, encoding="utf-8").stdout
    probes[name] = json.loads(txt)
probes["round 6 candidate"] = json.load(open(sys.argv[1]))
GUNCLAP = ("gunshot", "machine gun fire", "automatic fire", "gunfire")
for key, title in (("193426", "heavy rain storm"), ("194250", "dry storm")):
    reports, shift = calibrated_reports(U + key)
    ev = [json.loads(x) for x in open(f"{U}{key}/replay_ship0_s0.jsonl", encoding="utf-8")]
    near = [e for e in ev if any(-0.05 <= e["sample"] / 48000 - r[0] <= 0.15 for r in reports)]
    far = [e for e in ev if all(abs(e["sample"] / 48000 - r[0]) > 1.5 for r in reports)]
    shots_with = sum(any(-0.05 <= e["sample"] / 48000 - r[0] <= 0.15 for e in ev) for r in reports)
    print(f"\n{title}: {len(reports)} shots, {shots_with} with an event near; {len(near)} events near a shot, {len(far)} with no shot near; mean event level near {np.mean([e['level'] for e in near]):.1f} dB")
    by = {}
    for r in reports:
        k = (r[2], round(r[3] / 25) * 25)
        hit = any(-0.05 <= e["sample"] / 48000 - r[0] <= 0.15 for e in ev)
        by.setdefault(k, [0, 0])
        by[k][0] += hit
        by[k][1] += 1
    print("  shots with an event, by weapon and distance:", {f"{k[0]} {k[1]}m": f"{v[0]}/{v[1]}" for k, v in sorted(by.items())})
    print("  CLAP's own word near a shot:", {w: sum(e["clap"] == w for e in near) for w in sorted({e["clap"] for e in near}, key=lambda w: -sum(e["clap"] == w for e in near))[:6]})
    for name, p in probes.items():
        A, B, C = np.array(p["Weights"], dtype=np.float64), np.array(p["Bias"], dtype=np.float64), p["Classes"]
        def counts(g):
            c = {}
            for e in g:
                k = C[int(np.argmax(A @ np.array(e["embed"], dtype=np.float64) + B))]
                c[k] = c.get(k, 0) + 1
            return c
        cn, cf = counts(near), counts(far)
        print(f"  {name:<26} near a shot: gunfire {cn.get('gunfire', 0)}/{len(near)} {dict(sorted(cn.items(), key=lambda x: -x[1]))}; no shot near, gunfire {cf.get('gunfire', 0)}/{len(far)}")
