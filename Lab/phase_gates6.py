"""
Round 6's registered gates (2ae76ef), from the app's replay output. Usage: python phase_gates6.py <tag> <runs dir> [--gunfire-only]
Also prints each dry storm under any tag, so the shipped probe can be read before the candidate.
"""
import csv, datetime, json, sys
import numpy as np

TAG, ROOT = sys.argv[1], sys.argv[2].rstrip("/") + "/sniper_20261104_"
DRY_CH, DRY_NA, ATT_NA = "195113", "200652", "202233"
GUN_RAIN, GUN_DRY = "193426", "194250"
GUN = ("gunshot", "automatic fire")
GH = ("gas strike", "heli crash")
FALSE = GH + ("sleigh crash", "thunder", "infected attack")


def load(key):
    run = ROOT + key
    m = json.load(open(run + "/audio.json"))
    st = datetime.datetime.fromisoformat(m["start_utc"])
    s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
    world = [(int(e["game_ms"]) / 1000 + off - s, e["event"], float(e["distance"])) for e in csv.DictReader(open(run + "/world.csv"))]
    env = [(int(e["game_ms"]) / 1000 + off - s, float(e["infected_nearest"])) for e in csv.DictReader(open(run + "/env.csv"))]
    ev = [json.loads(x) for x in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8")]
    return world, np.array([x[0] for x in env]), np.array([x[1] for x in env]), ev


def words(g):
    w = {}
    for e in g:
        w[e["label"]] = w.get(e["label"], 0) + 1
    return dict(sorted(w.items(), key=lambda x: -x[1])[:8])


def rate(g, names):
    return sum(e["label"] in names for e in g)


def split(key):
    world, et, near, ev = load(key)
    attack, storm, other = [], [], []
    for e in ev:
        t = e["sample"] / 48000.0
        d = near[np.argmin(np.abs(et - t))]
        phase, since = "before", 0
        for i, (when, name, dist) in enumerate(world):
            end = world[i + 1][0] if i + 1 < len(world) else float("inf")
            if when <= t < end:
                phase, since = name, t - when
        if phase == "attack":
            if since >= 3 and 0 <= d < 3:
                attack.append(e)
            continue
        if key == "085810" and t >= 655:
            attack.append(e)
            continue
        if phase == "storm":
            storm.append(e)
        elif (phase == "rest" and since < 3) or (phase in ("clear_after", "rain") and since < 20 and key != "085810"):
            continue
        else:
            other.append(dict(e, _p=f"{key} {phase}"))
    return attack, storm, other


def check_attack(title, g, minimum=1):
    n = max(len(g), 1)
    ok = len(g) >= minimum and rate(g, ("infected attack",)) / n >= 0.7 and rate(g, ("thunder",)) / n <= 0.05 and rate(g, GH) / n <= 0.05 and rate(g, GUN) == 0
    print(f"{'PASS' if ok else 'FAIL'} {title}: {len(g)} events; infected attack {rate(g, ('infected attack',))} ({100 * rate(g, ('infected attack',)) / n:.1f}%, need 70); thunder {rate(g, ('thunder',))}; gas or heli {rate(g, GH)}; gunfire {rate(g, GUN)}; {words(g)}")
    return ok


def check_storm(title, g):
    n = max(len(g), 1)
    ok = len(g) > 0 and rate(g, ("thunder",)) / n >= 0.8 and rate(g, GH) / n <= 0.05 and rate(g, ("infected attack",)) / n <= 0.05 and rate(g, GUN) == 0
    print(f"{'PASS' if ok else 'FAIL'} {title}: {len(g)} events; thunder {rate(g, ('thunder',))} ({100 * rate(g, ('thunder',)) / n:.1f}%, need 80); gas or heli {rate(g, GH)}; infected attack {rate(g, ('infected attack',))}; gunfire {rate(g, GUN)}; {words(g)}")
    return ok


def check_gunfire(title, key):
    sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
    from lab_clock import calibrated_reports
    reports, info = calibrated_reports(ROOT + key)
    ev = [json.loads(x) for x in open(f"{ROOT}{key}/replay{TAG}_s0.jsonl", encoding="utf-8")]
    near, far = [], []
    for e in ev:
        t = e["sample"] / 48000.0
        if any(-0.05 <= t - r[0] <= 0.15 for r in reports):
            near.append(e)
        elif all(abs(t - r[0]) > 1.5 for r in reports):
            far.append(e)
    n = max(len(near), 1)
    ok = len(near) > 0 and rate(near, GUN) / n >= 0.9
    print(f"{'PASS' if ok else 'FAIL'} {title}: {len(reports)} shots, clock {info}; {len(near)} events near a shot; gunfire {rate(near, GUN)} ({100 * rate(near, GUN) / n:.1f}%, need 90); {words(near)}")
    print(f"     no shot within 1.5 s: {len(far)} events; called gunfire {rate(far, GUN)}; {words(far)}")
    return ok


if "--gunfire-only" in sys.argv:
    check_gunfire("Gunfire in a rain storm", GUN_RAIN)
    check_gunfire("Gunfire in a dry storm", GUN_DRY)
    sys.exit(0)

parts = {k: split(k) for k in (DRY_CH, DRY_NA, ATT_NA, "144529", "145757", "152249", "151013", "121745", "135013", "085810", "110232")}
results = [
    check_gunfire("Gunfire in a rain storm", GUN_RAIN),
    check_gunfire("Gunfire in a dry storm", GUN_DRY),
    check_storm("Dry storm, Chernarus, fresh", parts[DRY_CH][1]),
    check_storm("Dry storm, Nasdara, fresh", parts[DRY_NA][1]),
    check_attack("Attacks, Nasdara, fresh", parts[ATT_NA][0], 100),
    check_storm("Storm, Nasdara mode 21, fresh", parts[ATT_NA][1]),
    check_attack("Rain attacks, 144529 + 145757", parts["144529"][0] + parts["145757"][0], 100),
    check_attack("Clear attacks, 152249", parts["152249"][0]),
    check_attack("Clear attacks, 121745", parts["121745"][0]),
    check_attack("Natural attack, 085810", parts["085810"][0]),
    check_storm("Rain storms, all", sum((parts[k][1] for k in ("121745", "135013", "085810", "110232", "144529", "145757", "151013", "152249")), [])),
]
others = sum((parts[k][2] for k in parts), [])
sites = {}
for key in ("133615", "160723", "162444"):
    world, et, near, ev = load(key)
    moves = [w for w in world if w[1] == "site"]
    strikes = json.load(open(ROOT + key + "/strikes.json"))
    for e in ev:
        t = e["sample"] / 48000.0
        if t < moves[0][0] or any(when <= t < when + 8 for when, n, x in moves):
            continue
        d = near[np.argmin(np.abs(et - t))]
        if 0 <= d < 3 or any(at - 2 <= t <= at + 12 for ncc, at in strikes):
            continue
        site = max(x for when, n, x in moves if when <= t)
        others.append(dict(e, _p=f"survey {key}"))
        sites.setdefault((key, site), []).append(e)
n = max(len(others), 1)
ok = rate(others, FALSE) / n <= 0.05 and rate(others, GUN) == 0
print(f"{'PASS' if ok else 'FAIL'} Every other phase + surveys: {len(others)} events; event words {rate(others, FALSE)} ({100 * rate(others, FALSE) / n:.1f}%, at most 5); gunfire {rate(others, GUN)}; {words(others)}")
results.append(ok)
big = [(rate(g, FALSE) / len(g), k, len(g)) for k, g in sites.items() if len(g) >= 5]
worst = max(big, default=(0, ("-", -1), 0))
print(f"{'PASS' if worst[0] <= 0.20 else 'FAIL'} Worst survey site with 5+ events: {100 * worst[0]:.0f}% ({worst[1]}, {worst[2]} events; at most 20)")
results.append(worst[0] <= 0.20)
by = {}
for e in others:
    by.setdefault(e["_p"], []).append(e)
print("False event words by phase:")
for k, g in sorted(by.items()):
    if rate(g, FALSE):
        print(f"  {k:<20} {rate(g, FALSE)}/{len(g)} {[(e['label'], round(e['sample'] / 48000, 1)) for e in g if e['label'] in FALSE][:8]}")
print("ALL PHASE GATES PASS" if all(results) else "PHASE GATES FAILED")
