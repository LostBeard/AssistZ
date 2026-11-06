"""
Embeds the survey's labelled frames with DINOv2 in the real browser path (the app's /vision page, lab Chrome on port 9333) and saves the vectors. Frames stay on the work disk.
Usage: python vision_embed.py <frames dir> <base url> <out npz> <model small|base> <tiles 0|1> [max frames]
"""
import asyncio, base64, csv, json, os, sys, time, urllib.request
import numpy as np
import websockets

PORT = 9333
DIR, BASE, OUT, MODEL, TILES = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
LIMIT = int(sys.argv[6]) if len(sys.argv) > 6 else 0


def http(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=10) as r:
        return json.loads(r.read())


class Tab:
    def __init__(self, ws_url):
        self.ws_url, self.n, self.pending = ws_url, 0, {}
        self.rows, self.notes, self.done = [], [], False

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
                if text.startswith("[AssistZ] dino file="):
                    p = dict(x.split("=", 1) for x in text[len("[AssistZ] dino "):].split(" "))
                    self.rows.append((p["file"], p["kind"], float(p["ms"]), np.frombuffer(base64.b64decode(p["v"]), dtype=np.float32)))
                    if len(self.rows) % 200 == 0:
                        print(len(self.rows), "vectors", flush=True)
                elif "[AssistZ] dino" in text or "rror" in text:
                    self.notes.append(text)
                    print(text[:200], flush=True)
                    if "dino done" in text or "dino failed" in text:
                        self.done = True
            elif m.get("method") == "Runtime.exceptionThrown":
                print("EXCEPTION", json.dumps(m["params"]["exceptionDetails"])[:400], flush=True)

    async def call(self, method, **params):
        self.n += 1
        fut = asyncio.get_event_loop().create_future()
        self.pending[self.n] = fut
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        return await asyncio.wait_for(fut, 120)


async def main():
    rows = list(csv.DictReader(open(os.path.join(DIR, "manifest.csv"))))
    if LIMIT:
        rows = rows[:: max(1, len(rows) // LIMIT)][:LIMIT]
    files = [os.path.join(DIR, r["frame"]) for r in rows]
    print(len(files), "frames", flush=True)
    target = [t for t in http("/json") if t.get("type") == "page"][0]
    tab = Tab(target["webSocketDebuggerUrl"])
    await tab.open()
    await tab.call("Runtime.enable")
    await tab.call("Page.enable")
    await tab.call("DOM.enable")
    await tab.call("Page.navigate", url=f"{BASE}/vision?model={MODEL}&tiles={TILES}")
    for _ in range(240):
        await asyncio.sleep(0.5)
        r = await tab.call("Runtime.evaluate", expression="!!document.getElementById('az-frames')", returnByValue=True)
        if r.get("result", {}).get("result", {}).get("value"):
            break
    else:
        raise SystemExit("the page never showed its file input")
    await asyncio.sleep(1)
    doc = await tab.call("DOM.getDocument")
    node = await tab.call("DOM.querySelector", nodeId=doc["result"]["root"]["nodeId"], selector="#az-frames")
    await tab.call("DOM.setFileInputFiles", nodeId=node["result"]["nodeId"], files=files)
    t0 = time.time()
    while not tab.done:
        await asyncio.sleep(1)
        if time.time() - t0 > 7200:
            raise SystemExit("timed out: " + str(tab.notes[-3:]))
    print(f"{len(tab.rows)} vectors in {time.time() - t0:.0f} s", flush=True)
    if not tab.rows:
        raise SystemExit("no vectors: " + str(tab.notes[-3:]))
    ms = np.array([r[2] for r in tab.rows])
    kinds = sorted({r[1] for r in tab.rows})
    by_kind = {k: [] for k in kinds}
    names = {k: [] for k in kinds}
    times = {k: [] for k in kinds}
    for f, k, t, v in tab.rows:
        by_kind[k].append(v)
        names[k].append(f)
        times[k].append(t)
    np.savez(OUT, **{f"v_{k}": np.stack(by_kind[k]) for k in kinds}, **{f"n_{k}": np.array(names[k]) for k in kinds})
    for k in kinds:
        t = np.array(times[k][2:])
        print(f"{k}: {len(times[k])} vectors, first {times[k][0]:.0f} ms, then median {np.median(t):.1f} p90 {np.percentile(t, 90):.1f} max {t.max():.1f} ms; dim {len(by_kind[k][0])}, norm {np.linalg.norm(by_kind[k][0]):.3f}", flush=True)


asyncio.run(main())
