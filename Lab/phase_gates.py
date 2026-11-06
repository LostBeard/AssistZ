"""
Round 4's registered gates (82c59a4), from the app's replay output.
Rain attacks, fresh mode 21 runs (144529, 145757) together, at least 100 events: at least 70% infected attack; at most 5% thunder; at most 5% gas or heli; 0 gunfire.
Clear attacks, fresh mode 17 (152249, and 151013 which barely attacked) and 121745, each; and 085810's natural attack: the same.
Storms, all of those plus 110232 and 135013: at least 80% thunder; at most 5% gas or heli; at most 5% infected attack; 0 gunfire.
Every other phase of all of those, plus survey 133615: at most 5% gas, heli, thunder or infected attack; 0 gunfire; no survey site over 20%.
Usage: python phase_gates.py <tag> <runs dir>
"""
import csv, datetime, json, sys
import numpy as np

TAG, ROOT = sys.argv[1], sys.argv[2].rstrip("/") + "/sniper_"
R = {k: ROOT + "20261104_" + k for k in ("144529", "145757", "152249", "151013", "121745", "135013", "085810", "110232", "133615")}
WORLD = ("gas strike", "heli crash", "sleigh crash")
GUN = ("gunshot", "automatic fire")
FALSE = WORLD + ("thunder", "infected attack")


def load(run):
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
    return dict(sorted(w.items(), key=lambda x: -x[1])[:7])


def rate(g, names):
    return sum(e["label"] in names for e in g)


def report(title, g, want, need, limits, minimum=1):
    n = max(len(g), 1)
    ok = (rate(g, (want,)) / n >= need if want else True) and len(g) >= minimum
    line = f"{title}: {len(g)} events"
    if minimum > 1:
        line += f" (at least {minimum})"
    if want:
        line += f"; {want} {rate(g, (want,))} ({100 * rate(g, (want,)) / n:.1f}%, need {100 * need:.0f})"
    for names, lim, label in limits:
        c = rate(g, names)
        passed = c / n <= lim
        ok = ok and passed
        line += f"; {label} {c} ({100 * c / n:.1f}%, at most {100 * lim:.0f})"
    print(("PASS " if ok else "FAIL ") + line)
    print("     ", words(g))
    return ok


attacks = {}
storm_all, other_all = [], []
for key in ("144529", "145757", "152249", "151013", "121745", "135013", "085810", "110232"):
    run = R[key]
    world, et, near, ev = load(run)
    attacks[key] = []
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
                attacks[key].append(e)
            continue
        if key == "085810" and t >= 655:
            continue
        if phase == "storm":
            storm_all.append(e)
        elif (phase == "rest" and since < 3) or (phase in ("clear_after", "rain") and since < 20 and key != "085810"):
            continue
        else:
            other_all.append(dict(e, _p=f"{key} {phase}"))
world, et, near, ev = load(R["133615"])
sites = [w for w in world if w[1] == "site"]
strikes = json.load(open(R["133615"] + "/strikes.json"))
by_site = {}
for e in ev:
    t = e["sample"] / 48000.0
    if t < sites[0][0] or any(when <= t < when + 8 for when, n, x in sites):
        continue
    d = near[np.argmin(np.abs(et - t))]
    if 0 <= d < 3 or any(at - 2 <= t <= at + 12 for ncc, at in strikes):
        continue
    site = max(x for when, n, x in sites if when <= t)
    other_all.append(dict(e, _p=f"survey site {site:.0f}"))
    by_site.setdefault(site, []).append(e)
nat = [e for e in load(R["085810"])[3] if e["sample"] / 48000.0 >= 655]
limits = [(("thunder",), 0.05, "thunder"), (("gas strike", "heli crash"), 0.05, "gas or heli"), (GUN, 0.0, "gunfire")]
results = [
    report("Rain attacks, fresh 144529 + 145757", attacks["144529"] + attacks["145757"], "infected attack", 0.7, limits, 100),
    report("Clear attacks, fresh 152249", attacks["152249"], "infected attack", 0.7, limits),
    report("Clear attacks, 121745", attacks["121745"], "infected attack", 0.7, limits),
    report("Attack, 085810 natural (rain)", nat, "infected attack", 0.7, limits),
    report("Storms, all", storm_all, "thunder", 0.8, [(("gas strike", "heli crash"), 0.05, "gas or heli"), (("infected attack",), 0.05, "infected attack"), (GUN, 0.0, "gunfire")]),
    report("Every other phase + survey", other_all, None, 0, [(FALSE, 0.05, "false event words"), (GUN, 0.0, "gunfire")]),
]
print(f"(151013 barely attacked: {len(attacks['151013'])} scored events {words(attacks['151013'])}; 135013: {len(attacks['135013'])})")
worst = max(((rate(g, FALSE) / len(g), s, len(g)) for s, g in by_site.items()), default=(0, -1, 0))
site_ok = worst[0] <= 0.20
print(("PASS " if site_ok else "FAIL ") + f"Worst survey site: site {worst[1]:.0f}, {100 * worst[0]:.1f}% of {worst[2]} events (at most 20)")
results.append(site_ok)
print("Other phases, words by phase:")
by = {}
for e in other_all:
    by.setdefault(e["_p"], []).append(e)
for k, g in sorted(by.items()):
    print(f"  {k:<20} {len(g):3d} {words(g)}")
print("ALL PHASE GATES PASS" if all(results) else "PHASE GATES FAILED")
