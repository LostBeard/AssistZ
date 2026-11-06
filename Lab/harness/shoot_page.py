"""Screenshot of one app page in a lab Chrome. Usage: shoot_page.py <url> <out.png> [width height]"""
import asyncio, base64, json, os, subprocess, sys, time, urllib.request
import websockets
url, out = sys.argv[1], sys.argv[2]
W, H = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (760, 1000)
PORT = 9333
p = subprocess.Popen([r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"--user-data-dir=L:\AssistZLabWork\chrome-profile", f"--remote-debugging-port={PORT}", "--no-first-run", "--no-default-browser-check", "--new-window", url])


def http(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=5) as r:
        return json.loads(r.read())


async def main():
    t0 = time.time()
    page = None
    while page is None and time.time() - t0 < 30:
        try:
            page = next((t for t in http("/json") if t["type"] == "page" and "127.0.0.1" in t["url"]), None)
        except Exception:
            pass
        await asyncio.sleep(0.5)
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=50_000_000) as ws:
        i = 0

        async def call(method, **params):
            nonlocal i
            i += 1
            await ws.send(json.dumps({"id": i, "method": method, "params": params}))
            while True:
                m = json.loads(await ws.recv())
                if m.get("id") == i:
                    return m

        await call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=2, mobile=False)
        for _ in range(60):
            r = await call("Runtime.evaluate", expression="document.querySelectorAll('.hud-mark').length", returnByValue=True)
            if r["result"]["result"].get("value", 0) > 0:
                break
            await asyncio.sleep(0.5)
        await asyncio.sleep(1)
        shot = await call("Page.captureScreenshot", format="png")
        open(out, "wb").write(base64.b64decode(shot["result"]["data"]))
        print("saved", out, "marks", r["result"]["result"].get("value"))


try:
    asyncio.run(main())
finally:
    subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
