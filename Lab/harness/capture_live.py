"""
Drive AssistZ in the lab Chrome (CDP :9333) against a live source picked by Chrome's auto-select flag.
Prints telemetry, every detected sound event with its wall-clock time (UTC, for alignment with the DayZ pose CSV),
and saves screenshots. Usage: python capture_live.py <base_url> <seconds> <shot_prefix> [console file] [name]; name = load the sound namer first.
"""
import asyncio, json, os, re, sys, time, base64, datetime, urllib.request
import websockets

PORT = 9333
BASE, SECONDS, SHOT = sys.argv[1], float(sys.argv[2]), sys.argv[3]
CONSOLE_OUT = sys.argv[4] if len(sys.argv) > 4 else None
NAME_SOUNDS = len(sys.argv) > 5 and sys.argv[5] == "name"

def http(path, method="GET"):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", method=method)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())

async def main():
    page = next(t for t in http("/json") if t["type"] == "page")
    ws = await websockets.connect(page["webSocketDebuggerUrl"], max_size=50_000_000)
    n = 0; pending = {}; console = []
    async def reader():
        async for msg in ws:
            m = json.loads(msg)
            if "id" in m and m["id"] in pending: pending.pop(m["id"]).set_result(m)
            elif m.get("method") == "Runtime.consoleAPICalled":
                text = " ".join(str(a.get("value", a.get("description", ""))) for a in m["params"]["args"])
                console.append((datetime.datetime.now(datetime.timezone.utc), text))
            elif m.get("method") == "Runtime.exceptionThrown":
                console.append((datetime.datetime.now(datetime.timezone.utc), "EXCEPTION " + json.dumps(m["params"]["exceptionDetails"])[:400]))
    asyncio.create_task(reader())
    async def call(method, **params):
        nonlocal n
        n += 1
        fut = asyncio.get_event_loop().create_future(); pending[n] = fut
        await ws.send(json.dumps({"id": n, "method": method, "params": params}))
        return await asyncio.wait_for(fut, 30)
    async def ev(expr, gesture=False):
        r = await call("Runtime.evaluate", expression=expr, userGesture=gesture, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("result", {}).get("value")
    await call("Runtime.enable")
    btn = "[...document.querySelectorAll('button')].find(b=>b.textContent.includes('Share window'))"
    booted = False
    for _ in range(120):
        if await ev(f"!!({btn})"):
            booted = True
            break
        await asyncio.sleep(0.5)
    if not booted:
        # The app never rendered: print why (Blazor's error UI and the console) and stop, rather than "capturing" nothing.
        print("APP DID NOT BOOT:", await ev("document.body.innerText.slice(0,300)"))
        for ts, text in console[:20]: print("  console:", text[:300])
        open(os.path.join(os.path.dirname(SHOT) or ".", "APP_DID_NOT_BOOT"), "w").write("see capture output")
        return
    print("APP READY")
    if NAME_SOUNDS:
        named = "[...document.querySelectorAll('button')].find(b=>b.textContent.includes('%s'))"
        await ev(f"({named % 'Name sounds'}).click()", gesture=True)
        t1 = time.time()
        while not await ev(f"!!({named % 'Naming sounds'})"):
            assert time.time() - t1 < 600, "the namer never loaded"
            await asyncio.sleep(0.5)
        print(f"namer ready in {time.time() - t1:.0f} s")
    await ev(f"({btn}).click()", gesture=True)
    t0 = time.time()
    shots = 0
    while time.time() - t0 < SECONDS:
        await asyncio.sleep(min(10, SECONDS - (time.time() - t0)))
        shot = await call("Page.captureScreenshot", format="png")
        open(f"{SHOT}_{shots}.png", "wb").write(base64.b64decode(shot["result"]["data"])); shots += 1
    print("=== status:", await ev("document.querySelector('.status')?.innerText"))
    print("=== telemetry\n" + str(await ev("document.querySelector('.telemetry')?.innerText")).replace("\n\n", "\n"))
    print("=== toast:", await ev("document.querySelector('.toast')?.innerText"))
    if CONSOLE_OUT:
        with open(CONSOLE_OUT, "w", encoding="utf-8") as f:
            for ts, text in console: f.write(ts.isoformat() + " " + text + "\n")
    events = 0
    for ts, text in console:
        if "[AssistZ] event" in text:
            events += 1
            print(ts.strftime("%H:%M:%S.%f")[:-3], text.replace("[AssistZ] event ", ""))
        elif ("AssistZ" in text and "] yaw " not in text) or "EXCEPTION" in text: print("console:", text[:300])
    print(f"=== {events} events in {SECONDS:.0f} s, {shots} screenshots")

asyncio.run(main())
