"""The own sounds filter live: what the app named in each driver phase, by wall clock (a name comes about 1 s after its sound). Limits as on replay: movement at most 5% gunfire or infected, at least 70% own. Usage: own_live_score.py <console> <drive json> <app log>"""
import collections, datetime, json, re, sys
pat = re.compile(r"\[AssistZ\] label utc=(\d+):(\d+):([\d.]+) end=\d+ label=(.+?) raw=")
drive = json.load(open(sys.argv[2]))
print("driver:", "stopped: " + drive["stopped"] if drive.get("stopped") else f"{len(drive['phases'])} phases")
ph = [(p["name"], datetime.datetime.fromisoformat(p["start_utc"]), datetime.datetime.fromisoformat(p["end_utc"])) for p in drive["phases"]]
by = collections.defaultdict(collections.Counter)
n = 0
for line in open(sys.argv[1], encoding="utf-8"):
    m = pat.search(line)
    if not m:
        continue
    n += 1
    stamp = datetime.datetime.fromisoformat(line.split(" ")[0]) - datetime.timedelta(seconds=1.0)
    name = next((a for a, lo, hi in ph if lo <= stamp < hi), "outside")
    by[name][m[4]] += 1
print(f"{n} names")
for a, _, _ in ph:
    print(f"  {a:<13} {sum(by[a].values()):>4}  {dict(by[a].most_common(5))}")
print(f"  {'outside':<13} {sum(by['outside'].values()):>4}  {dict(by['outside'].most_common(5))}")
move = collections.Counter()
for a, _, _ in ph:
    if a != "fire" and not a.startswith("still"):
        move += by[a]
total = sum(move.values())
bad = move["gunshot"] + move["automatic fire"] + move["infected attack"]
own = move["own move"]
print(f"{'PASS' if bad <= 0.05 * total else 'FAIL'} movement named gunfire or infected {bad}/{total} (at most 5%)")
print(f"{'PASS' if own >= 0.70 * total else 'FAIL'} movement named own {own}/{total} (at least 70%)")
print("fire:", dict(by["fire"].most_common(5)))
log = open(sys.argv[3], encoding="utf-8", errors="replace").read()
for key in ("Stream", "Audio lost", "Skipped"):
    m = re.search(key + r"\s+([^\r\n]+)", log)
    print(f"on screen {key}: {m[1] if m else '?'}")
m = re.search(r"=== toast: ([^\r\n]*)", log)
print("toast:", m[1] if m else "?")
