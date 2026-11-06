"""
Score a replay (verify_replay.py .jsonl) against its run's shot log, without a browser. Shots at or past the recording's end are
not counted: the log can run longer than the recording.
Usage: python score_replay.py <run dir> <replay.jsonl> [...]
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_clock import calibrated_reports

GUNFIRE = ("gunshot", "automatic fire")
RUN = sys.argv[1]
reports, shift = calibrated_reports(RUN)
duration = json.load(open(RUN + "/audio.json"))["frames"] / 48000.0
inside = [r for r in reports if 0 <= r[0] < duration]
print(f"{os.path.basename(RUN)}: clock {shift * 1000:+.0f} ms; {len(reports)} shots, {len(inside)} inside the recording")
for path in sys.argv[2:]:
    events = [json.loads(x) for x in open(path, encoding="utf-8")]
    times = [e["sample"] / 48000.0 for e in events]
    near = gunfire = mode_ok = 0
    for e, t in zip(events, times):
        r = min(inside, key=lambda x: abs(x[0] - t))
        if abs(r[0] - t) < 0.25:
            near += 1
            if e["label"] in GUNFIRE:
                gunfire += 1
                mode_ok += (e["label"] == "automatic fire") == (r[2] == "AugShort")
    with_event = sum(any(-0.05 <= t - r[0] <= 0.15 for t in times) for r in inside)
    named = sum(any(-0.05 <= t - r[0] <= 0.15 and e["label"] in GUNFIRE for e, t in zip(events, times)) for r in inside)
    print(f"  {os.path.basename(path)}: {len(events)} events; near a report {near}, gunfire {gunfire}, mode {mode_ok}/{gunfire}; shots with an event {with_event}/{len(inside)}, named gunfire {named}/{len(inside)}")
