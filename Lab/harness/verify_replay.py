"""
Deterministic CLAP labels on a lab recording: drives the app's ?replay=1 path over CDP (lab Chrome, port 9333), one tab per
skip value, and writes every event with all its CLAP scores to <run>/replay<tag>_s<skip>.jsonl. Positions are exact file samples,
and the shot log is calibrated on the recording, so the score below has no capture clock in it.
Usage: python verify_replay.py <run dir> <base url> <skips, comma separated> <tag> [extra query]
"""
import asyncio, json, os, re, sys, time, urllib.request
import numpy as np
import websockets
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lab_clock import calibrated_reports

PORT = 9333
RUN, BASE, SKIPS, TAG = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")], sys.argv[4]
QUERY = sys.argv[5] if len(sys.argv) > 5 else ""
GUNFIRE = ("gunshot", "automatic fire")


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
                self.console.append(" ".join(str(a.get("value", a.get("description", ""))) for a in m["params"]["args"]))
            elif m.get("method") == "Runtime.exceptionThrown":
                self.console.append("EXCEPTION " + json.dumps(m["params"]["exceptionDetails"])[:400])
    async def call(self, method, **params):
        self.n += 1
        fut = asyncio.get_event_loop().create_future()
        self.pending[self.n] = fut
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        return await asyncio.wait_for(fut, 60)
    async def eval(self, expr, gesture=False):
        r = await self.call("Runtime.evaluate", expression=expr, userGesture=gesture, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("result", {}).get("value")


def parse(console, skip):
    labels, scores, embeds = {}, {}, {}
    for text in console:
        m = re.search(r"\[AssistZ\] label .*end=(\d+) label=(.+?) raw=(.+?) clap=(.+?) p=([\d.]+) ms=([\d.]+) az=(-?[\d.]+) level=(-?[\d.]+)", text)
        if m:
            labels[int(m[1])] = dict(sample=int(m[1]) + skip, label=m[2], raw=m[3], clap=m[4], p=float(m[5]), ms=float(m[6]), az=float(m[7]), level=float(m[8]))
            dist = re.search(r" dist=([\d.]+)", text)
            if dist:
                labels[int(m[1])]["dist"] = float(dist[1])
            gun = re.search(r" gun=(\S+) family=(\S+)", text)
            if gun:
                if gun[1] != "-":
                    labels[int(m[1])]["gun"] = gun[1]
                if gun[2] != "-":
                    labels[int(m[1])]["family"] = gun[2]
        m = re.search(r"\[AssistZ\] relabel end=(\d+) label=(.+)$", text)
        if m and int(m[1]) in labels:
            labels[int(m[1])]["label"] = m[2].strip()
        m = re.search(r"\[AssistZ\] scores end=(\d+) clip=(\d+) v=([^ ]+)", text)
        if m:
            scores[int(m[1])] = (int(m[2]) + skip, [float(x) for x in m[3].strip(",").split(",")])
        m = re.search(r"\[AssistZ\] embed end=(\d+) v=([^ ]+)", text)
        if m:
            embeds[int(m[1])] = [float(x) for x in m[2].strip(",").split(",")]
    for end, e in labels.items():
        e["clip"], e["scores"] = scores[end]
        if end in embeds:
            e["embed"] = embeds[end]
    return [labels[k] for k in sorted(labels)]


async def one(browser, skip, reports):
    await browser.send(json.dumps({"id": 1, "method": "Target.createTarget", "params": {"url": f"{BASE}/?lab=1&replay=1&skip={skip}{QUERY}"}}))
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
    t0 = time.time()
    while not await app.eval(f"!!({btn % 'Naming sounds'})"):
        assert not await app.eval("document.querySelector('.telemetry')?.innerText.includes('failed')"), "CLAP load failed"
        assert time.time() - t0 < 1800, "CLAP never loaded"
        await asyncio.sleep(0.5)
    doc = await app.call("DOM.getDocument")
    node = await app.call("DOM.querySelector", nodeId=doc["result"]["root"]["nodeId"], selector="#az-file")
    await app.call("DOM.setFileInputFiles", nodeId=node["result"]["nodeId"], files=[RUN.replace("/", "\\") + "\\audio.wav"])
    t0 = time.time()
    while not any("replay done" in x or "replay failed" in x for x in app.console):
        assert time.time() - t0 < 900, "replay never finished: " + str(app.console[-3:])
        await asyncio.sleep(0.5)
    await asyncio.sleep(0.5)
    summary = next(x for x in app.console if "replay done" in x or "replay failed" in x)
    print(f"--- skip {skip}: {summary}")
    for x in app.console:
        if "warmup" in x or "EXCEPTION" in x or "rror" in x: print("  console:", x[:300])
    assert "replay done" in summary, summary
    events = parse(app.console, skip)
    mem = [x for x in app.console if x.startswith("[AssistZ] mem")]
    if mem:
        with open(f"{RUN}/replay{TAG}_mem.log", "w", encoding="utf-8") as f:
            f.write("\n".join(mem) + "\n")
    path = f"{RUN}/replay{TAG}_s{skip}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for e in events: f.write(json.dumps(e) + "\n")
    # Events: nearest report within 250 ms (verify_labels.py's measure). Shots: named gunfire by an event ending -50..+150 ms
    # around the report; an event ends up to one window + hop after the sound starts.
    if not reports:
        print("  no shot log: nothing to score")
    near = gunfire = mode_ok = 0
    for e in events if reports else []:
        t = e["sample"] / 48000.0
        r = min(reports, key=lambda x: abs(x[0] - t))
        if abs(r[0] - t) < 0.25:
            near += 1
            if e["label"] in GUNFIRE:
                gunfire += 1
                mode_ok += (e["label"] == "automatic fire") == (r[2] == "AugShort")
    heard = sum(any(-0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15 and e["label"] in GUNFIRE for e in events) for r in reports)
    any_event = sum(any(-0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15 for e in events) for r in reports)
    ms = [e["ms"] for e in events] or [0.0]
    print(f"  {len(events)} labelled -> {path}")
    print(f"  events near a report {near}, named gunfire {gunfire}, fire mode right {mode_ok}/{gunfire}")
    print(f"  shots with an event {any_event}/{len(reports)}, named gunfire {heard}/{len(reports)}; CLAP ms median {np.median(ms):.1f} max {max(ms):.1f}")
    # Close our socket first: a socket the browser closes under a running reader is an unretrieved exception.
    app.reader.cancel()
    await app.ws.close()
    await browser.send(json.dumps({"id": 2, "method": "Target.closeTarget", "params": {"targetId": target_id}}))
    while json.loads(await browser.recv()).get("id") != 2: pass


async def main():
    reports, shift = calibrated_reports(RUN)
    print(f"shot log clock shift on the recording {shift * 1000:+.0f} ms; {len(reports)} shots")
    browser = await websockets.connect(http("/json/version")["webSocketDebuggerUrl"], max_size=10_000_000)
    # about:blank stays open so closing a tab never closes the browser.
    for skip in SKIPS:
        await one(browser, skip, reports)

asyncio.run(main())
