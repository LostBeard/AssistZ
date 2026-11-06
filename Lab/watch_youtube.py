"""
Watch-along smoke test with nobody at the keyboard: in the lab Chrome (CDP :9333, started with
--auto-select-tab-capture-source-by-title="AssistZ Source" and --autoplay-policy=no-user-gesture-required) it opens a
YouTube search, plays the first video result, retitles that tab "AssistZ Source" (kept by a MutationObserver), opens
AssistZ in lab mode and shares the tab - tab capture includes the tab's audio. Saves screenshots, the AssistZ
console and a summary.
Usage: python watch_youtube.py <assistz_url> <query> <seconds> <out_dir>
"""
import asyncio, json, sys, time, base64, datetime, os, urllib.parse, urllib.request
import websockets

PORT = 9333
APP, QUERY, SECONDS, OUT = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
os.makedirs(OUT, exist_ok=True)

def http(path, method="GET"):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", method=method)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

class Tab:
    def __init__(self, target):
        self.url = target["webSocketDebuggerUrl"]; self.n = 0; self.pending = {}; self.console = []
    async def open(self):
        self.ws = await websockets.connect(self.url, max_size=50_000_000)
        asyncio.create_task(self._read())
        await self.call("Runtime.enable"); await self.call("Page.enable")
    async def _read(self):
        async for msg in self.ws:
            m = json.loads(msg)
            if "id" in m and m["id"] in self.pending: self.pending.pop(m["id"]).set_result(m)
            elif m.get("method") == "Runtime.consoleAPICalled":
                text = " ".join(str(a.get("value", a.get("description", ""))) for a in m["params"]["args"])
                self.console.append((datetime.datetime.now(datetime.timezone.utc), text))
    async def call(self, method, **params):
        self.n += 1
        fut = asyncio.get_event_loop().create_future(); self.pending[self.n] = fut
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        return await asyncio.wait_for(fut, 60)
    async def ev(self, expr, gesture=False):
        r = await self.call("Runtime.evaluate", expression=expr, userGesture=gesture, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("result", {}).get("value")
    async def wait(self, expr, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                v = await self.ev(expr)
                if v: return v
            except Exception: pass
            await asyncio.sleep(0.5)
        return None

async def main():
    yt = Tab(next(t for t in http("/json") if t["type"] == "page"))
    await yt.open()
    await yt.call("Page.navigate", url="https://www.youtube.com/results?search_query=" + urllib.parse.quote(QUERY))
    href = await yt.wait("[...document.querySelectorAll('a#video-title, a#video-title-link')].map(a=>a.href).find(h=>h.includes('/watch'))", 45)
    if not href:
        print("no video result found"); return
    await yt.call("Page.navigate", url=href)
    playing = await yt.wait("(()=>{const v=document.querySelector('video'); return v && !v.paused && v.currentTime > 1 ? v.currentTime : 0})()", 60)
    title = await yt.ev("document.title")
    print("video:", href, "|", title, "| playing at", playing)
    # The player fills the tab, so the capture is the game picture and not the YouTube page around it (a static page
    # around the video drags the view-turn tracker toward 0). requestFullscreen did not survive the AssistZ window taking
    # focus, so the harness pins the player with CSS instead; the page is only a test source.
    await yt.ev("""(()=>{const s=document.createElement('style'); s.textContent=
      '#movie_player{position:fixed!important;inset:0!important;width:100vw!important;height:100vh!important;z-index:2147483647!important;background:#000}'
      + ' #movie_player video{width:100vw!important;height:100vh!important;left:0!important;top:0!important;object-fit:contain}'
      + ' .ytp-chrome-bottom,.ytp-chrome-top,.ytp-gradient-bottom,.ytp-gradient-top{display:none!important}';
      document.head.appendChild(s); window.dispatchEvent(new Event('resize')); return true})()""")
    await asyncio.sleep(1.5)
    # Keep the title stable so the tab auto-select flag finds it.
    await yt.ev("document.title='AssistZ Source'; new MutationObserver(()=>{ if (document.title!=='AssistZ Source') document.title='AssistZ Source'; }).observe(document.querySelector('title'), {childList:true})")

    # AssistZ in its OWN window: a second tab in the same window hides the YouTube tab, and Chrome stops decoding video
    # in hidden tabs (the first run captured a black player with live audio).
    browser = await websockets.connect(http("/json/version")["webSocketDebuggerUrl"], max_size=10_000_000)
    await browser.send(json.dumps({"id": 1, "method": "Target.createTarget", "params": {"url": APP, "newWindow": True}}))
    target_id = None
    while target_id is None:
        m = json.loads(await browser.recv())
        if m.get("id") == 1: target_id = m["result"]["targetId"]
    await asyncio.sleep(1.0)
    app = Tab(next(t for t in http("/json") if t["id"] == target_id))
    await app.open()
    btn = "[...document.querySelectorAll('button')].find(b=>b.textContent.includes('Share window'))"
    if not await app.wait(f"!!({btn})", 60):
        print("AssistZ never rendered"); return
    await app.ev(f"({btn}).click()", gesture=True)
    t0 = time.time(); shots = 0
    while time.time() - t0 < SECONDS:
        await asyncio.sleep(min(15, SECONDS - (time.time() - t0)))
        s = await app.call("Page.captureScreenshot", format="png")
        open(os.path.join(OUT, f"shot_{shots}.png"), "wb").write(base64.b64decode(s["result"]["data"])); shots += 1
    telemetry = await app.ev("document.querySelector('.telemetry')?.innerText")
    toast = await app.ev("document.querySelector('.toast')?.innerText")
    status = await app.ev("document.querySelector('.status')?.innerText")
    with open(os.path.join(OUT, "console.txt"), "w", encoding="utf-8") as f:
        for ts, text in app.console: f.write(ts.isoformat() + " " + text + "\n")
    events = [t for _, t in app.console if "[AssistZ] event" in t]
    yaws = [t for _, t in app.console if "] yaw " in t]
    summary = {"video": href, "title": title, "status": status, "toast": toast, "telemetry": telemetry,
               "events": len(events), "yaw_samples": len(yaws), "seconds": SECONDS}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    for t in events[:25]: print(t)

asyncio.run(main())
