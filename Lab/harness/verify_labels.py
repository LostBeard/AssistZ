"""
CLAP sound naming, end to end on a DayZ lab recording with ground truth. Drives the lab Chrome (own profile, port 9333):
opens the app, clicks "Name sounds" (first time: 620 MB of towers into OPFS), feeds a lab run's loopback recording to
"Open recording", and collects the "[AssistZ] label ..." console lines.
Ground truth: the run's client shot log (aishots_to_shots.py -> shots_ai.csv) put on the recording's clock (pose.csv
game_ms -> UTC, audio.json start). An event's place in the recording is its EndSample / 48000 (the recording plays from
0 and the capture starts with it), so each label is matched to the nearest shot report (fire time + distance / 330 m/s).
Usage: python verify_labels.py <run dir> <base url> <seconds> <console out>
"""
import asyncio, csv, datetime, json, os, re, sys, time, urllib.request
import numpy as np
import websockets
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lab_clock import calibrated_reports

PORT = 9333
RUN, BASE, SECONDS, CONSOLE_OUT = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
QUERY = sys.argv[5] if len(sys.argv) > 5 else ""   # extra query, e.g. "&clap=larger"

def http(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=10) as r:
        return json.loads(r.read())

class Tab:
    def __init__(self, ws_url):
        self.ws_url, self.n, self.pending, self.console = ws_url, 0, {}, []
    async def open(self):
        self.ws = await websockets.connect(self.ws_url, max_size=50_000_000)
        self.reader = asyncio.create_task(self._read())
    async def _read(self):
        async for msg in self.ws:
            m = json.loads(msg)
            if "id" in m and m["id"] in self.pending:
                self.pending.pop(m["id"]).set_result(m)
            elif m.get("method") == "Runtime.consoleAPICalled":
                self.console.append((time.time(), " ".join(str(a.get("value", a.get("description", ""))) for a in m["params"]["args"])))
            elif m.get("method") == "Runtime.exceptionThrown":
                self.console.append((time.time(), "EXCEPTION " + json.dumps(m["params"]["exceptionDetails"])[:400]))
    async def call(self, method, **params):
        self.n += 1
        fut = asyncio.get_event_loop().create_future()
        self.pending[self.n] = fut
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        return await asyncio.wait_for(fut, 60)
    async def eval(self, expr, gesture=False):
        r = await self.call("Runtime.evaluate", expression=expr, userGesture=gesture, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("result", {}).get("value")

def recording_clock():
    """(game seconds -> seconds into the recording) for this run's server clock."""
    meta = json.load(open(RUN + "/audio.json"))
    st = datetime.datetime.fromisoformat(meta["start_utc"])
    start_s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(RUN + "/pose.csv")))
    g = np.array([int(r["game_ms"]) / 1000 for r in pose])
    u = np.array([sum(int(x) * k for x, k in zip(r["utc"].split(":"), (3600, 60, 1))) for r in pose], dtype=float)
    offset = np.median(u + 0.5 - g)
    return lambda game_s: game_s + offset - start_s

def ai_track():
    """The raider's logged pose (lab mode 6 aipose.csv) on the recording clock: [(t, ai, speed m/s, dist m, state)]."""
    import os
    if not os.path.exists(RUN + "/aipose.csv"): return []
    clock = recording_clock()
    return [(clock(int(r["game_ms"]) / 1000), r["ai"].replace("eAI_Survivor", ""), float(r["speed"]), float(r["dist"]), r["state"])
            for r in csv.DictReader(open(RUN + "/aipose.csv"))]

def shot_reports():
    """Report arrival times (s into the recording) of the run's shots, with shooter, distance and miss."""
    meta = json.load(open(RUN + "/audio.json"))
    st = datetime.datetime.fromisoformat(meta["start_utc"])
    start_s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(RUN + "/pose.csv")))
    g = np.array([int(r["game_ms"]) / 1000 for r in pose])
    u = np.array([sum(int(x) * k for x, k in zip(r["utc"].split(":"), (3600, 60, 1))) for r in pose], dtype=float)
    offset = np.median(u + 0.5 - g)
    out = []
    for r in csv.DictReader(open(RUN + "/shots_ai.csv")):
        t = int(r["game_ms"]) / 1000 + offset - start_s + float(r["distance"]) / 330.0
        out.append((t, r["shooter"].replace("eAI_Survivor", ""), r["weapon"], float(r["distance"]), float(r["miss"])))
    return out

async def main():
    browser = await websockets.connect(http("/json/version")["webSocketDebuggerUrl"], max_size=10_000_000)
    await browser.send(json.dumps({"id": 1, "method": "Target.createTarget", "params": {"url": f"{BASE}/?lab=1{QUERY}"}}))
    target_id = None
    while target_id is None:
        msg = json.loads(await browser.recv())
        if msg.get("id") == 1: target_id = msg["result"]["targetId"]
    await asyncio.sleep(1.0)
    app = Tab(next(t for t in http("/json") if t["id"] == target_id)["webSocketDebuggerUrl"])
    await app.open()
    await app.call("Runtime.enable")
    await app.call("DOM.enable")
    btn = "[...document.querySelectorAll('button')].find(b=>b.textContent.includes('%s'))"
    t0 = time.time()
    while not await app.eval(f"!!({btn % 'Name sounds'})"):
        assert time.time() - t0 < 60, "app never rendered"
        await asyncio.sleep(0.5)
    await app.eval(f"({btn % 'Name sounds'}).click()", gesture=True)
    t_load = time.time()
    last = None
    while not await app.eval(f"!!({btn % 'Naming sounds'})"):
        status = await app.eval("[...document.querySelectorAll('button')].map(b=>b.textContent.trim()).join(' | ')")
        if status != last: print(f"  {time.time() - t_load:6.1f} s  {status}"); last = status
        failed = await app.eval("document.querySelector('.telemetry')?.innerText.includes('failed')")
        assert not failed, "CLAP load failed: " + str(await app.eval("document.querySelector('.telemetry p.muted')?.innerText"))
        assert time.time() - t_load < 1800, "CLAP never loaded"
        await asyncio.sleep(1.0)
    print(f"=== CLAP ready after {time.time() - t_load:.1f} s")
    doc = await app.call("DOM.getDocument")
    node = await app.call("DOM.querySelector", nodeId=doc["result"]["root"]["nodeId"], selector="#az-file")
    wav = RUN.replace("/", "\\") + "\\audio.wav"
    await app.call("DOM.setFileInputFiles", nodeId=node["result"]["nodeId"], files=[wav])
    t_play = time.time()
    await asyncio.sleep(SECONDS)
    stats = await app.eval("document.querySelector('.telemetry')?.innerText")
    err = await app.eval("document.querySelector('.toast')?.innerText")
    print("=== stats\n" + str(stats)); print("=== error:", err)
    with open(CONSOLE_OUT, "w", encoding="utf-8") as f:
        for ts, text in app.console: f.write(f"{ts:.3f} {text}\n")
    labels = []
    relabel = {}
    for ts, text in app.console:
        m = re.search(r"\[AssistZ\] relabel end=(\d+) label=(.+)$", text)
        if m: relabel[int(m[1])] = m[2].strip()
    for ts, text in app.console:
        m = re.search(r"\[AssistZ\] label .*end=(\d+) label=(.+?) (?:raw=.+? )?(?:clap=(.+?) )?p=([\d.]+) ms=([\d.]+)", text)
        if m: labels.append((int(m[1]) / 48000.0, relabel.get(int(m[1]), m[2]), float(m[4]), float(m[5]), m[3] or m[2]))
        elif "EXCEPTION" in text or "rror" in text: print("console:", text[:300])
    # Sample 0 is wherever the audio track happened to start (measured +145..+215 ms between runs on one GPU, which moved this
    # score from 31/43 to 41/52), so positions go on the file's clock with the app's own playclock lines, and the shot log
    # goes on the file's clock with lab_clock. Events past the first pass of the looped recording are left out.
    clock = [float(m[2]) - int(m[1]) / 48000.0 for _, text in app.console if (m := re.search(r"\[AssistZ\] playclock end=(\d+) ct=([\d.]+)", text))]
    duration = json.load(open(RUN + "/audio.json"))["frames"] / 48000.0
    first_pass = [c for c in clock if c > -1.0 and c < 5.0]
    playclock = float(np.median(first_pass)) if first_pass else 0.0
    print(f"=== playclock: {len(clock)} lines, offset {playclock * 1000:+.0f} ms (first pass spread {(max(first_pass) - min(first_pass)) * 1000 if first_pass else 0:.0f} ms)")
    labels = [(l[0] + playclock, *l[1:]) for l in labels if l[0] + playclock < duration]
    reports, shot_shift = calibrated_reports(RUN)
    print(f"=== shot log clock shift on the recording {shot_shift * 1000:+.0f} ms")
    track = ai_track()
    moving_steps = steps_total = 0
    print(f"=== {len(labels)} labelled events in {SECONDS:.0f} s; {len(reports)} shots in the run's log")
    hits = 0
    # Fire mode truth from the weapon: the AUG is automatic, the SVD and the Scout fire single shots.
    mode_ok = mode_n = 0
    for pos, label, p, ms, clap in labels:
        near = min(reports, key=lambda r: abs(r[0] - pos), default=None)
        d = pos - near[0] if near else 99
        tag = f"shot {near[1]} {near[2]} {near[3]:.0f} m miss {near[4]:.1f} m ({d * 1000:+.0f} ms)" if near and abs(d) < 0.25 else ""
        if tag:
            hits += label in ("gunshot", "automatic fire")
            if label in ("gunshot", "automatic fire"):
                mode_n += 1
                mode_ok += (label == "automatic fire") == (near[2] == "AugShort")
        if track and not tag:
            # Footsteps reach the listener ~dist/330 s after the step; the pose sample nearest that time says what the AI did.
            a = min(track, key=lambda r: abs(r[0] + r[3] / 330.0 - pos))
            if abs(a[0] + a[3] / 330.0 - pos) < 0.5:
                tag = f"AI {a[1]} {a[2]:.1f} m/s at {a[3]:.0f} m, {a[4]}"
                if label == "footsteps":
                    steps_total += 1
                    moving_steps += a[2] > 0.5
        print(f"  {pos:7.2f} s  {label:<16} {clap:<17} {p:5.3f}  {ms:5.1f} ms  {tag}")
    shot_events = sum(1 for pos, *_ in labels if any(abs(r[0] - pos) < 0.25 for r in reports))
    times = [l[3] for l in labels]
    print(f"=== events within 250 ms of a shot report: {shot_events}, of them named gunfire: {hits}")
    if track: print(f"=== footsteps labels while the raider was moving (> 0.5 m/s): {moving_steps}/{steps_total}")
    print(f"=== fire mode right (automatic = AUG, single = SVD/Scout): {mode_ok}/{mode_n}")
    if times: print(f"=== CLAP ms median {np.median(times):.1f} max {max(times):.1f}")

asyncio.run(main())
