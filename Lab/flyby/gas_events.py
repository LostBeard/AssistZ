"""
Do the low rumbles line up with the game's gas strikes? For each recording: the rumble (2 s bins, level over -30 dB, centroid under 300 Hz) and the time the server log shows a ContaminatedArea_Dynamic zone being created, both on the recording's clock.
Server script logs carry local time; recordings store UTC. Reported as found; a run with neither is the control.
"""
import datetime, glob, json, os, re
import numpy as np

RUNS = r"D:\users\tj\Projects\AssistZ\Lab\runs"
LOGS = r"D:\users\tj\Projects\AssistZ\Lab\profiles\server"
logs = []
for f in sorted(glob.glob(LOGS + r"\script_2026-*.log")):
    m = re.search(r"script_(\d{4}-\d\d-\d\d)_(\d\d)-(\d\d)-(\d\d)", f)
    head = open(f, encoding="utf-8", errors="replace").read(600)
    u = re.search(r"\((\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d) UTC\)", head)
    started = re.search(r"started at (\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)", head)
    if not (u and started):
        continue
    start_utc = datetime.datetime.strptime(u[1] + " " + u[2], "%Y-%m-%d %H:%M:%S")
    start_local = datetime.datetime.strptime(started[1] + " " + started[2], "%Y-%m-%d %H:%M:%S")
    logs.append((start_utc, start_local, f))
print(len(logs), "server logs usable; UTC offset", logs[0][0] - logs[0][1] if logs else None)
rows = []
for run in sorted(glob.glob(RUNS + r"\sniper_2026110[34]_*")):
    if not os.path.exists(run + "/audio.json") or not os.path.exists(run + "/audio.npy"):
        continue
    meta = json.load(open(run + "/audio.json"))
    st = datetime.datetime.fromisoformat(meta["start_utc"]).replace(tzinfo=None)
    dur = meta["frames"] / 48000
    a = np.load(run + "/audio.npy")
    rumble = []
    for i in range(0, int(dur) - 1, 2):
        s = np.asarray(a[i * 48000: (i + 2) * 48000], dtype=np.float64)
        spec = np.abs(np.fft.rfft(s.mean(axis=1))) ** 2
        f = np.fft.rfftfreq(len(s), 1 / 48000)
        if 20 * np.log10(np.sqrt((s ** 2).mean()) + 1e-9) > -30 and (spec * f).sum() / (spec.sum() + 1e-12) < 300:
            rumble.append(i)
    log = [l for l in logs if l[0] <= st + datetime.timedelta(seconds=60) and l[0] > st - datetime.timedelta(minutes=8)]
    zones = []
    if log:
        start_utc, start_local, path = log[-1]
        seen = set()
        for line in open(path, encoding="utf-8", errors="replace"):
            if "ContaminatedArea_Dynamic" not in line or "InitZoneServ" not in line:
                continue
            t = re.match(r"\s*(\d\d):(\d\d):(\d\d)", line)
            if t:
                when = start_local.replace(hour=int(t[1]), minute=int(t[2]), second=int(t[3])) + (start_utc - start_local)
                sec = (when - st).total_seconds()
                if -60 <= sec <= dur + 60 and int(sec) not in seen:
                    seen.add(int(sec))
                    zones.append(int(sec))
    rows.append((os.path.basename(run)[-15:], int(dur), rumble, zones, bool(log)))
print(f"{'recording':<16} {'s':>4}  {'rumble bins (s)':<36} zone created (s)")
for name, dur, rumble, zones, has in rows:
    print(f"{name:<16} {dur:>4}  {str(rumble[:9]):<36} {zones[:6] if has else 'no server log kept'}")
with_log = [r for r in rows if r[4]]
both = [r for r in with_log if r[2] and r[3]]
print(f"\nRecordings with a server log: {len(with_log)}. Rumble and zone: {len(both)}; rumble, no zone: {sum(bool(r[2]) and not r[3] for r in with_log)}; zone, no rumble: {sum(bool(r[3]) and not r[2] for r in with_log)}; neither: {sum(not r[2] and not r[3] for r in with_log)}")
gaps = [z - r[2][0] for r in both for z in r[3][:1]]
print("zone time minus first rumble bin (s):", gaps)
