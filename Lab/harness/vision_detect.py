"""
Runs the library's YOLOv8 on frames in the real browser path (the app's /vision?model=yolo page, lab Chrome on port 9333) and saves what it finds. Frames stay on the work disk.
Usage: python vision_detect.py <frames dir> <base url> <out json> <tiles 0|1> [every nth frame]
"""
import asyncio, csv, json, os, sys, time, urllib.request
import numpy as np
import websockets
PORT = int(os.environ.get("VISION_PORT", "9333"))
DIR, BASE, OUT, TILES = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
NTH = int(sys.argv[5]) if len(sys.argv) > 5 else 1


def http(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=10) as r:
        return json.loads(r.read())


async def main():
    rows = list(csv.DictReader(open(os.path.join(DIR, "manifest.csv"))))[::NTH]
    files = [os.path.join(DIR, r["frame"]) for r in rows]
    print(len(files), "frames", flush=True)
    target = [t for t in http("/json") if t.get("type") == "page"][0]
    ws = await websockets.connect(target["webSocketDebuggerUrl"], max_size=50_000_000)
    n, pending, out, done, notes = 0, {}, [], [False], []

    async def reader():
        async for msg in ws:
            m = json.loads(msg)
            if "id" in m and m["id"] in pending:
                pending.pop(m["id"]).set_result(m)
            elif m.get("method") == "Runtime.consoleAPICalled":
                text = " ".join(str(a.get("value", a.get("description", ""))) for a in m["params"]["args"])
                if text.startswith("[AssistZ] yolo file="):
                    parts = text[len("[AssistZ] yolo "):].split(" ")
                    p = dict(x.split("=", 1) for x in parts)
                    found = []
                    for item in filter(None, p.get("found", "").split(";")):
                        a = item.split(":")
                        found.append(dict(label=a[0], conf=float(a[1]), x=float(a[2]), y=float(a[3]), w=float(a[4]), h=float(a[5])))
                    out.append(dict(file=p["file"], kind=p["kind"], ms=float(p["ms"]), found=found, stages=p.get("stages")))
                    if len(out) % 200 == 0:
                        print(len(out), "results", flush=True)
                elif "[AssistZ] yolo" in text or "rror" in text:
                    print(text[:240], flush=True)
                    notes.append(text)
                    if "yolo done" in text or "yolo failed" in text:
                        done[0] = True
            elif m.get("method") == "Runtime.exceptionThrown":
                print("EXCEPTION", json.dumps(m["params"]["exceptionDetails"])[:300], flush=True)

    task = asyncio.create_task(reader())

    async def call(method, **params):
        nonlocal n
        n += 1
        fut = asyncio.get_event_loop().create_future()
        pending[n] = fut
        await ws.send(json.dumps({"id": n, "method": method, "params": params}))
        return await asyncio.wait_for(fut, 120)

    await call("Runtime.enable")
    await call("Page.enable")
    await call("DOM.enable")
    # TILES can also carry the rest of the query; a query naming its own model wins.
    await call("Page.navigate", url=f"{BASE}/vision?tiles={TILES}" + ("" if "model=" in TILES else "&model=yolo"))
    for _ in range(240):
        await asyncio.sleep(0.5)
        r = await call("Runtime.evaluate", expression="!!document.getElementById('az-frames')", returnByValue=True)
        if r.get("result", {}).get("result", {}).get("value"):
            break
    else:
        raise SystemExit("the page never showed its file input")
    await asyncio.sleep(1)
    # Runtime.enable replays the tab's earlier console, so a second run in one tab read the first run's lines as its own.
    out.clear()
    done[0] = False
    doc = await call("DOM.getDocument")
    node = await call("DOM.querySelector", nodeId=doc["result"]["root"]["nodeId"], selector="#az-frames")
    await call("DOM.setFileInputFiles", nodeId=node["result"]["nodeId"], files=files)
    t0 = time.time()
    while not done[0]:
        await asyncio.sleep(1)
        if time.time() - t0 > 7200:
            raise SystemExit("timed out: " + str(notes[-3:]))
    json.dump(out, open(OUT, "w"))
    print(f"{len(out)} results in {time.time() - t0:.0f} s", flush=True)
    for k in sorted({o["kind"] for o in out}):
        t = np.array([o["ms"] for o in out if o["kind"] == k][2:])
        if t.size:
            print(f"{k}: median {np.median(t):.1f} p90 {np.percentile(t, 90):.1f} max {t.max():.1f} ms", flush=True)
    staged = np.array([[float(v) for v in o["stages"].split(",")] for o in out if o.get("stages")][20:])
    if staged.size:
        print("stages, median ms: " + ", ".join(f"{n} {v:.1f}" for n, v in zip(("image", "upload+preprocess", "run", "sync", "readback", "postprocess"), np.median(staged, axis=0))), flush=True)
    task.cancel()
    await ws.close()

asyncio.run(main())
