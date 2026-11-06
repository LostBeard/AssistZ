"""
Compare two verify_labels.py outputs on the same recording, event by event (cross-vendor parity, run-to-run control).

Why not compare the printed scores: an event's position is its EndSample / 48000, counted from the moment the capture
started, and playback of the recording starts at a slightly different time on each run. A constant offset of a few hundred
ms pushes shots out of verify_labels.py's +-250 ms window with no change in what was heard or how it was named. So B is
first put on A's clock (the shift that pairs the most events, measured here, not assumed), then labels and probabilities
are compared per event, and B is scored against the shot log on A's clock.
Positions come from the two-decimal labels file, so a score can differ by one from verify_labels.py's exact one when an
event sits on the 250 ms edge (run sniper_20261103_153510: 44 here, 43 exact).

Usage: python compare_labels.py <run dir> <labels A.txt> <labels B.txt>
"""
import csv, datetime, json, re, statistics, sys

RUN, FILE_A, FILE_B = sys.argv[1], sys.argv[2], sys.argv[3]
PAIR_TOL = 0.06        # s: two events are the same sound when within this after alignment (hop is 21.3 ms)
STARTUP = 5.0          # s: onsets at capture start are the detector settling, not sounds in the recording
GUNFIRE = ("gunshot", "automatic fire")
LINE = re.compile(r"^\s*(\d+\.\d+) s  (.{16}) (.{17}) ([\d.]+)\s+([\d.]+) ms")


def parse(path):
    events = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE.match(line)
            if m:
                events.append(dict(t=float(m[1]), label=m[2].strip(), clap=m[3].strip(), p=float(m[4]), ms=float(m[5])))
    return events


def shot_reports():
    """Report arrival times on the recording clock; the same calculation as verify_labels.py shot_reports."""
    meta = json.load(open(RUN + "/audio.json"))
    st = datetime.datetime.fromisoformat(meta["start_utc"])
    start_s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(RUN + "/pose.csv")))
    offsets = [sum(int(x) * k for x, k in zip(r["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(r["game_ms"]) / 1000 for r in pose]
    offset = statistics.median(offsets)
    return [(int(r["game_ms"]) / 1000 + offset - start_s + float(r["distance"]) / 330.0, r["weapon"])
            for r in csv.DictReader(open(RUN + "/shots_ai.csv"))]


def pair(a, b, shift):
    """One-to-one nearest pairs within PAIR_TOL, closest first, with b shifted by `shift`."""
    candidates = sorted((abs(x["t"] - (y["t"] + shift)), i, j) for i, x in enumerate(a) for j, y in enumerate(b)
                        if abs(x["t"] - (y["t"] + shift)) <= PAIR_TOL)
    used_a, used_b, pairs = set(), set(), []
    for _, i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); pairs.append((i, j))
    return sorted(pairs), used_a, used_b


def score(events, reports, shift=0.0):
    """verify_labels.py's score: events within 250 ms of a report; of them named gunfire; fire mode right."""
    near_n = gunfire = mode_ok = 0
    for e in events:
        t = e["t"] + shift
        r = min(reports, key=lambda x: abs(x[0] - t))
        if abs(r[0] - t) < 0.25:
            near_n += 1
            if e["label"] in GUNFIRE:
                gunfire += 1
                mode_ok += (e["label"] == "automatic fire") == (r[1] == "AugShort")
    return near_n, gunfire, mode_ok


all_a, all_b = parse(FILE_A), parse(FILE_B)
a = [e for e in all_a if e["t"] >= STARTUP]
b = [e for e in all_b if e["t"] >= STARTUP]
shifts = [s / 1000 for s in range(-600, 601)]
coarse = {s: len(pair(a, b, s)[0]) for s in shifts}
top = max(coarse.values())
# Among the shifts that pair the most events, the one with the smallest residual.
def spread(s):
    return statistics.median(abs(a[i]["t"] - (b[j]["t"] + s)) for i, j in pair(a, b, s)[0])
shift = min((s for s, n in coarse.items() if n == top), key=spread)
pairs, used_a, used_b = pair(a, b, shift)
residual = [a[i]["t"] - (b[j]["t"] + shift) for i, j in pairs]
dp_same = [abs(a[i]["p"] - b[j]["p"]) for i, j in pairs if a[i]["clap"] == b[j]["clap"]]
same_clap = sum(a[i]["clap"] == b[j]["clap"] for i, j in pairs)
same_family = sum((a[i]["label"] in GUNFIRE) == (b[j]["label"] in GUNFIRE) for i, j in pairs)
reports = shot_reports()

print(f"A: {FILE_A}: {len(all_a)} events ({len(a)} after {STARTUP:.0f} s)")
print(f"B: {FILE_B}: {len(all_b)} events ({len(b)} after {STARTUP:.0f} s)")
print(f"clock shift B -> A: {shift * 1000:+.0f} ms; paired {len(pairs)}; unpaired A {len(a) - len(used_a)}, B {len(b) - len(used_b)}")
print(f"pair residual after the shift: median |{statistics.median(abs(r) for r in residual) * 1000:.1f}| ms, max |{max(abs(r) for r in residual) * 1000:.1f}| ms")
print(f"same CLAP label: {same_clap}/{len(pairs)}; same gunfire or not: {same_family}/{len(pairs)}")
if dp_same:
    dp = sorted(dp_same)
    print(f"|p difference| where the label agrees ({len(dp)}): median {statistics.median(dp):.3f}, p90 {dp[int(len(dp) * 0.9)]:.3f}, max {dp[-1]:.3f}")
print(f"CLAP ms median: A {statistics.median(e['ms'] for e in all_a):.1f}, B {statistics.median(e['ms'] for e in all_b):.1f}")
na, ga, ma = score(all_a, reports)
print(f"score as run:       A near {na}, gunfire {ga}, mode {ma}/{ga}")
nb, gb, mb = score(all_b, reports)
print(f"score as run:       B near {nb}, gunfire {gb}, mode {mb}/{gb}")
nb, gb, mb = score(all_b, reports, shift)
print(f"score on A's clock: B near {nb}, gunfire {gb}, mode {mb}/{gb}")
print("disagreements (t on A's clock):")
for i, j in pairs:
    if a[i]["clap"] != b[j]["clap"]:
        print(f"  {a[i]['t']:7.2f} s  A {a[i]['clap']:<17} {a[i]['p']:.3f}   B {b[j]['clap']:<17} {b[j]['p']:.3f}")
for k, e in enumerate(a):
    if k not in used_a: print(f"  {e['t']:7.2f} s  A only: {e['clap']} {e['p']:.3f}")
for k, e in enumerate(b):
    if k not in used_b: print(f"  {e['t'] + shift:7.2f} s  B only: {e['clap']} {e['p']:.3f}")
