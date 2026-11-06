"""
AssistZ M0 end-to-end check. Drives a SEPARATE Chrome (own profile, own debug port) over CDP:
  tab 1 = /testsignal playing stereo pulses at pan -1, -0.5, 0, 0.5, 1
  tab 2 = / sharing tab 1 (auto-selected by title) -> GPU stereo analysis -> "[AssistZ] event az=..." console lines
Pass = events are detected and their azimuths cluster at -90, -45, 0, 45, 90 (+-10 deg).
Usage: python verify_m0.py <chrome_pid_file> <base_url> <seconds> <screenshot_png>
"""
import asyncio, json, re, sys, time, urllib.request, base64, collections
import websockets

PORT = 9333
BASE = sys.argv[2]
SECONDS = float(sys.argv[3])
SHOT = sys.argv[4]
# Optional: the full app URL (query string allowed, e.g. ?lab=1&spectral=0) and a file for the whole console.
APP_URL = sys.argv[5] if len(sys.argv) > 5 else f"{BASE}/"
CONSOLE_OUT = sys.argv[6] if len(sys.argv) > 6 else None

def http(path, method="GET"):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", method=method)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())

class Tab:
    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.n = 0
        self.pending = {}
        self.console = []
    async def open(self):
        self.ws = await websockets.connect(self.ws_url, max_size=50_000_000)
        self.reader = asyncio.create_task(self._read())
    async def _read(self):
        async for msg in self.ws:
            m = json.loads(msg)
            if "id" in m and m["id"] in self.pending:
                self.pending.pop(m["id"]).set_result(m)
            elif m.get("method") == "Runtime.consoleAPICalled":
                text = " ".join(str(a.get("value", a.get("description", ""))) for a in m["params"]["args"])
                self.console.append((time.time(), text))
            elif m.get("method") == "Runtime.exceptionThrown":
                self.console.append((time.time(), "EXCEPTION " + json.dumps(m["params"]["exceptionDetails"])[:400]))
    async def call(self, method, **params):
        self.n += 1
        fut = asyncio.get_event_loop().create_future()
        self.pending[self.n] = fut
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        return await asyncio.wait_for(fut, 30)
    async def eval(self, expr, gesture=False):
        r = await self.call("Runtime.evaluate", expression=expr, userGesture=gesture, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("result", {}).get("value")

async def wait_for(tab, expr, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if await tab.eval(expr):
            return True
        await asyncio.sleep(0.5)
    return False

async def main():
    targets = [t for t in http("/json") if t["type"] == "page"]
    sig = next(t for t in targets if "/testsignal" in t["url"])
    # Target.createTarget takes the URL as a JSON field: /json/new?<url> cut the URL at its first '&' (the first A/B run
    # lost &spectral=0 and measured "on" four times).
    browser = await websockets.connect(http("/json/version")["webSocketDebuggerUrl"], max_size=10_000_000)
    await browser.send(json.dumps({"id": 1, "method": "Target.createTarget", "params": {"url": APP_URL}}))
    target_id = None
    while target_id is None:
        msg = json.loads(await browser.recv())
        if msg.get("id") == 1: target_id = msg["result"]["targetId"]
    await asyncio.sleep(1.0)
    app_t = next(t for t in http("/json") if t["id"] == target_id)
    sig_tab, app = Tab(sig["webSocketDebuggerUrl"]), Tab(app_t["webSocketDebuggerUrl"])
    await sig_tab.open(); await app.open()
    for t in (sig_tab, app):
        await t.call("Runtime.enable")
    btn = "[...document.querySelectorAll('button')].find(b=>b.textContent.includes('%s'))"
    assert await wait_for(sig_tab, f"!!({btn % 'Play pulses'})"), "test signal page never rendered"
    await sig_tab.eval(f"({btn % 'Play pulses'}).click()", gesture=True)
    assert await wait_for(app, f"!!({btn % 'Share window'})"), "app page never rendered"
    await app.eval(f"({btn % 'Share window'}).click()", gesture=True)
    t_start = time.time()
    await asyncio.sleep(SECONDS)
    stats = await app.eval("document.querySelector('.telemetry')?.innerText")
    err = await app.eval("document.querySelector('.toast')?.innerText")
    shot = await app.call("Page.captureScreenshot", format="png")
    open(SHOT, "wb").write(base64.b64decode(shot["result"]["data"]))
    print("=== stats\n" + str(stats))
    print("=== error:", err)
    if CONSOLE_OUT:
        with open(CONSOLE_OUT, "w", encoding="utf-8") as f:
            for ts, text in app.console: f.write(f"{ts:.3f} {text}\n")
    events = []
    spectral = []   # (broadband az, spectral az) per event, when the build logs saz
    for ts, text in app.console:
        m = re.search(r"\[AssistZ\] event az=(-?[\d.]+) level=(-?[\d.]+)dB rise=(-?[\d.]+)dB lag=(-?\d+) corr=(-?[\d.]+)", text)
        if m:
            events.append(tuple(float(x) for x in m.groups()))
            s = re.search(r"saz=(-?[\d.]+) cen=([\d.]+) hf=([\d.]+)", text)
            if s: spectral.append((float(m[1]), float(s[1]), float(s[2]), float(s[3])))
        elif ("AssistZ" in text and not re.search(r"\] (abatch|yaw) ", text)) or "EXCEPTION" in text or "rror" in text: print("console:", text[:300])
    print(f"=== {len(events)} events in {SECONDS:.0f} s (expected ~{SECONDS / 0.6:.0f})")
    buckets = collections.Counter()
    off = []
    for az, lvl, rise, lag, corr in events:
        target = min((-90, -45, 0, 45, 90), key=lambda t: abs(t - az))
        if abs(target - az) <= 10: buckets[target] += 1
        else: off.append(az)
    print("=== azimuth buckets (+-10 deg):", dict(sorted(buckets.items())), " outside:", [round(a, 1) for a in off])
    print("=== lags seen:", sorted(collections.Counter(int(e[3]) for e in events).items()))
    for e in events[:12]: print("   az=%.1f level=%.1f rise=%.1f lag=%d corr=%.2f" % e)
    # ORDER: the pulses cycle -90 -> -45 -> 0 -> 45 -> 90 -> -90. The pan set is symmetric, so a left/right swap would
    # still fill every bucket; only the sequence catches it.
    cycle = [-90, -45, 0, 45, 90]
    snapped = [min(cycle, key=lambda t: abs(t - e[0])) for e in events]
    steps = list(zip(snapped, snapped[1:]))
    good = sum(1 for a, b in steps if cycle[(cycle.index(a) + 1) % 5] == b)
    print(f"=== order: {good}/{len(steps)} consecutive events step to the next pan in the cycle")
    rises = [e[2] for e in events]
    print(f"=== rise dB min/max: {min(rises, default=0):.1f} / {max(rises, default=0):.1f}")
    # SPECTRAL: the square-wave pulses are panned identically in every band, so the frontend's energy-weighted band ILD
    # must give the same azimuth as the broadband levels.
    spec_ok = True
    if spectral:
        diffs = [abs(a - s) for a, s, _, _ in spectral]
        print(f"=== spectral az vs broadband az: {len(spectral)} events, |diff| mean {sum(diffs) / len(diffs):.2f} max {max(diffs):.2f} deg;"
              f" centroid {min(c for *_, c, _ in spectral):.3f}..{max(c for *_, c, _ in spectral):.3f}")
        spec_ok = max(diffs) <= 5.0
    else:
        print("=== no spectral fields logged (older build)")
    ok = (len(events) >= SECONDS / 0.6 * 0.7 and len(buckets) == 5 and len(off) <= len(events) * 0.1 and not err
          and len(steps) > 0 and good >= len(steps) * 0.9 and max(rises, default=999) < 100 and spec_ok)
    print("RESULT:", "PASS" if ok else "FAIL")

asyncio.run(main())
