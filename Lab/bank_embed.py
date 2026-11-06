"""
CLAP embeddings of the sound bank's clips through the real browser path (/vision?model=bank, lab Chrome on port 9333), the same call the app makes. Each chunk writes emb_NNN.npy: clips x levels x 512. Restartable: chunks already done are skipped. Stays local.
Usage: python bank_embed.py <bank dir> <base url> [levels, default -6,-24] [first chunk] [last chunk]
"""
import asyncio, base64, glob, json, os, sys, time, urllib.request
import numpy as np
import websockets
PORT = int(os.environ.get("VISION_PORT", "9333"))
DIR, BASE = sys.argv[1], sys.argv[2]
LEVELS = sys.argv[3] if len(sys.argv) > 3 else "-6,-24"
FIRST = int(sys.argv[4]) if len(sys.argv) > 4 else 0
LAST = int(sys.argv[5]) if len(sys.argv) > 5 else 10 ** 6
NLEVELS = len(LEVELS.split(","))


def http(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=10) as r:
        return json.loads(r.read())


async def main():
    files = sorted(glob.glob(os.path.join(DIR, "clips_*.f32")))
    target = [t for t in http("/json") if t.get("type") == "page"][0]
    ws = await websockets.connect(target["webSocketDebuggerUrl"], max_size=200_000_000)
    n, pending = 0, {}
    state = dict(done=False, rows={}, loaded=False, failed=None)

    async def reader():
        async for msg in ws:
            m = json.loads(msg)
            if "id" in m and m["id"] in pending:
                pending.pop(m["id"]).set_result(m)
            elif m.get("method") == "Runtime.consoleAPICalled":
                text = " ".join(str(a.get("value", a.get("description", ""))) for a in m["params"]["args"])
                if text.startswith("[AssistZ] bank e "):
                    _, _, _, i, lv, b64 = text.split(" ")
                    state["rows"][(int(i), int(lv))] = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
                elif text.startswith("[AssistZ] bank loaded"):
                    state["loaded"] = True
                    print(text[:160], flush=True)
                elif text.startswith("[AssistZ] bank done"):
                    state["done"] = True
                elif text.startswith("[AssistZ] bank failed"):
                    state["failed"] = text
                    print(text[:300], flush=True)
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

    await call("Runtime.enable"); await call("Page.enable"); await call("DOM.enable")
    await call("Page.navigate", url=f"{BASE}/vision?model=bank&levels={LEVELS}")
    for _ in range(240):
        await asyncio.sleep(0.5)
        r = await call("Runtime.evaluate", expression="!!document.getElementById('az-frames')", returnByValue=True)
        if r.get("result", {}).get("result", {}).get("value"):
            break
    await asyncio.sleep(1)
    t0 = time.time()
    for path in files:
        k = int(os.path.basename(path)[6:9])
        out = os.path.join(DIR, f"emb_{k:03d}.npy")
        if k < FIRST or k > LAST or os.path.exists(out):
            continue
        clips = os.path.getsize(path) // (48000 * 4)
        state.update(done=False, rows={}, failed=None)
        doc = await call("DOM.getDocument")
        node = await call("DOM.querySelector", nodeId=doc["result"]["root"]["nodeId"], selector="#az-frames")
        await call("DOM.setFileInputFiles", nodeId=node["result"]["nodeId"], files=[path])
        c0 = time.time()
        while not state["done"] and not state["failed"]:
            await asyncio.sleep(0.5)
            if time.time() - c0 > 1800:
                raise SystemExit(f"chunk {k} timed out with {len(state['rows'])} rows")
        if state["failed"]:
            raise SystemExit(state["failed"])
        a = np.zeros((clips, NLEVELS, 512), dtype=np.float32)
        missing = 0
        for i in range(clips):
            for lv in range(NLEVELS):
                row = state["rows"].get((i, lv))
                if row is None:
                    missing += 1
                else:
                    a[i, lv] = row
        np.save(out, a)
        print(f"chunk {k}: {clips} clips, {len(state['rows'])} embeddings, missing {missing}, {time.time() - c0:.0f} s; {time.time() - t0:.0f} s in all", flush=True)
    task.cancel()
    await ws.close()
    print("EMBED DONE", flush=True)

asyncio.run(main())
