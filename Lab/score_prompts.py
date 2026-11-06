"""
Prompts only, against the game's logs, per recording and in total: for comparing clip normalization with no normalization and with the embedding probe.
Usage: python score_prompts.py <tag> <run> [...]    reads <run>/replay<tag>_s0.jsonl
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_clock import calibrated_reports

TAG, RUNS = sys.argv[1], sys.argv[2:]
GUN = ("gunshot", "automatic fire")
tg = ng = to = no = 0
for run in RUNS:
    reports, _ = calibrated_reports(run)
    g = o = cg = co = 0
    aug = caug = 0
    words = {}
    for line in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8"):
        e = json.loads(line)
        t = e["sample"] / 48000.0
        named = e["label"] in GUN
        near = [r for r in reports if -0.05 <= t - r[0] <= 0.15]
        if near:
            g += 1
            cg += named
            if near[0][2] == "AugShort" and near[0][3] > 100:
                aug += 1
                caug += named
        elif all(abs(t - r[0]) > 1.5 for r in reports):
            o += 1
            co += named
            words[e["clap"]] = words.get(e["clap"], 0) + 1
    extra = f"; 150 m AUG {caug}/{aug}" if aug else ""
    print(f"  {os.path.basename(run.rstrip('/'))[-15:]}: gunfire named gunfire {cg}/{g}{extra}; not gunfire called gunfire {co}/{o}; words {dict(sorted(words.items(), key=lambda x: -x[1])[:4])}")
    tg += g; ng += cg; to += o; no += co
print(f"  TOTAL: gunfire named gunfire {ng}/{tg}; not gunfire called gunfire {no}/{to}")
