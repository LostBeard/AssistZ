"""
Drives the lab character so it makes its own sounds, with nobody at the keyboard (lab mode 32): stands, walks, jogs, sprints, crouches, crawls, jumps, fires. Every phase's UTC is logged. Safety: before every step the foreground window must belong to the DayZ client PID recorded by launch_dayzdiag.ps1; otherwise it lets go of every key and stops.
Usage: python drive_own.py <dayzdiag.pids.json> <out.json>                (lab mode 32: own sounds)
       python drive_own.py <dayzdiag.pids.json> <out.json> see <seconds>  (lab mode 34: keeps moving and sweeping the view while frames are saved, for video on a moving player)
       python drive_own.py <dayzdiag.pids.json> <out.json> screens <seconds>  (lab mode 34 with -DieAt: opens the pause menu, the map and the inventory, goes prone and crawls: what does video make of them?)
"""
import atexit, ctypes, ctypes.wintypes as wt, datetime, json, sys, time

user32 = ctypes.windll.user32
raw = open(sys.argv[1], "rb").read()
# launch_dayzdiag.ps1 writes UTF-16 from a PowerShell started with -NoProfile, UTF-8 otherwise.
pids = json.loads(raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"))
client_pid = int(pids["client"])
out_path = sys.argv[2]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD), ("time", wt.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD), ("dwFlags", wt.DWORD), ("time", wt.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("pad", ctypes.c_byte * 32)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wt.DWORD), ("u", _U)]


KEYUP, SCANCODE = 0x0002, 0x0008
# Scan codes: the game reads these, not virtual keys.
W, S, SHIFT, CTRL, C, X, V, TAB, ESC, M = 0x11, 0x1F, 0x2A, 0x1D, 0x2E, 0x2D, 0x2F, 0x0F, 0x01, 0x32
held = set()
mouse_held = set()
LEFT, RIGHT = (0x0002, 0x0004), (0x0008, 0x0010)


def key(scan, down):
    inp = INPUT(type=1)
    inp.ki = KEYBDINPUT(0, scan, SCANCODE | (0 if down else KEYUP), 0, 0)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    (held.add if down else held.discard)(scan)


def look(dx):
    inp = INPUT(type=0)
    inp.mi = MOUSEINPUT(dx, 0, 0, 0x0001, 0, 0)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def button(which, down):
    inp = INPUT(type=0)
    inp.mi = MOUSEINPUT(0, 0, 0, which[0 if down else 1], 0, 0)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    (mouse_held.add if down else mouse_held.discard)(which)


def let_go():
    for s in list(held):
        key(s, False)
    for b in list(mouse_held):
        button(b, False)


atexit.register(let_go)


def foreground_pid():
    p = wt.DWORD()
    user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), ctypes.byref(p))
    return p.value


def window_of(pid):
    found = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        p = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
        if p.value == pid and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True
    user32.EnumWindows(cb, 0)
    return found[0] if found else None


def focus(hwnd):
    user32.ShowWindow(hwnd, 9)
    fg = user32.GetForegroundWindow()
    cur = ctypes.windll.kernel32.GetCurrentThreadId()
    fg_thread = user32.GetWindowThreadProcessId(fg, None)
    user32.AttachThreadInput(cur, fg_thread, True)
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    user32.AttachThreadInput(cur, fg_thread, False)
    time.sleep(0.5)
    return foreground_pid() == client_pid


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


log = {"client_pid": client_pid, "phases": []}


def stop(reason):
    let_go()
    log["stopped"] = reason
    json.dump(log, open(out_path, "w"), indent=1)
    sys.exit("stopped: " + reason)


def wait(seconds):
    end = time.time() + seconds
    while time.time() < end:
        if foreground_pid() != client_pid:
            stop("DayZ client lost the foreground")
        time.sleep(0.05)


def phase(name, seconds, keys=(), reverse_each=0):
    """Holds keys for the phase; with reverse_each, swaps W and S so the character stays near where it started."""
    seg = {"name": name, "start_utc": utc()}
    for k in keys:
        key(k, True)
    if reverse_each:
        left = seconds
        forward = True
        while left > 0:
            step = min(reverse_each, left)
            wait(step)
            left -= step
            if left > 0:
                key(W if forward else S, False)
                time.sleep(0.15)
                key(S if forward else W, True)
                forward = not forward
        key(W, False)
        key(S, False)
    else:
        wait(seconds)
    for k in keys:
        key(k, False)
    seg["end_utc"] = utc()
    log["phases"].append(seg)
    json.dump(log, open(out_path, "w"), indent=1)


hwnd = window_of(client_pid)
if not hwnd:
    sys.exit(f"no visible window for DayZ client PID {client_pid}")
if not focus(hwnd):
    sys.exit(f"could not bring DayZ client to the foreground (foreground PID {foreground_pid()})")
rect = wt.RECT()
user32.GetClientRect(hwnd, ctypes.byref(rect))
pt = wt.POINT((rect.right - rect.left) // 2, (rect.bottom - rect.top) // 2)
user32.ClientToScreen(hwnd, ctypes.byref(pt))
user32.SetCursorPos(pt.x, pt.y)
time.sleep(0.2)
# A click inside the window captures the mouse: with a lowered rifle it does nothing else.
button(LEFT, True); time.sleep(0.05); button(LEFT, False)
time.sleep(1.0)



def sweep(name, seconds, counts_per_second, keys=(), reverse_each=0):
    """Turns the view one way for half the time and back for the other half, with keys held. Reverses W and S as phase does."""
    seg = {"name": name, "start_utc": utc(), "counts_per_second": counts_per_second}
    for k in keys:
        key(k, True)
    start = time.time()
    last_swap = start
    forward = True
    while time.time() - start < seconds:
        if foreground_pid() != client_pid:
            stop("DayZ client lost the foreground")
        direction = 1 if (time.time() - start) < seconds / 2 else -1
        look(int(direction * counts_per_second * 0.05))
        if reverse_each and time.time() - last_swap >= reverse_each:
            key(W if forward else S, False)
            key(S if forward else W, True)
            forward = not forward
            last_swap = time.time()
        time.sleep(0.05)
    for k in (W, S) + tuple(keys):
        key(k, False)
    seg["end_utc"] = utc()
    log["phases"].append(seg)
    json.dump(log, open(out_path, "w"), indent=1)


def toggle(name, scan, seconds, close=None):
    """Opens a screen with a key, holds it, and closes it."""
    seg = {"name": name, "start_utc": utc()}
    key(scan, True); time.sleep(0.1); key(scan, False)
    wait(seconds)
    seg["end_utc"] = utc(); log["phases"].append(seg)
    closer = close if close is not None else scan
    key(closer, True); time.sleep(0.1); key(closer, False)
    wait(1.5)
    json.dump(log, open(out_path, "w"), indent=1)


if len(sys.argv) > 3 and sys.argv[3] == "screens":
    until = time.time() + float(sys.argv[4])
    while time.time() < until:
        phase("still", 12)
        toggle("pause_menu", ESC, 8)
        phase("still", 6)
        toggle("map", M, 8)
        phase("still", 6)
        toggle("inventory", TAB, 8)
        phase("still", 6)
        seg = {"name": "prone_down", "start_utc": utc()}
        key(X, True); time.sleep(0.1); key(X, False)
        wait(3)
        seg["end_utc"] = utc(); log["phases"].append(seg)
        phase("crawl", 12, (W,), reverse_each=6)
        key(X, True); time.sleep(0.1); key(X, False)
        wait(3)
    let_go()
    json.dump(log, open(out_path, "w"), indent=1)
    print(f"done: {len(log['phases'])} phases -> {out_path}")
    sys.exit(0)

if len(sys.argv) > 3 and sys.argv[3] == "see":
    until = time.time() + float(sys.argv[4])
    while time.time() < until:
        phase("still", 8)
        phase("walk", 24, (CTRL, W), reverse_each=6)
        sweep("sweep", 16, 60)
        phase("jog", 24, (W,), reverse_each=6)
        sweep("walk_sweep", 24, 40, (CTRL, W), reverse_each=6)
        phase("sprint", 16, (SHIFT, W), reverse_each=4)
        seg = {"name": "inventory", "start_utc": utc()}
        key(TAB, True); time.sleep(0.1); key(TAB, False)
        wait(8)
        seg["end_utc"] = utc(); log["phases"].append(seg)
        key(TAB, True); time.sleep(0.1); key(TAB, False)
        wait(1)
    let_go()
    json.dump(log, open(out_path, "w"), indent=1)
    print(f"done: {len(log['phases'])} phases -> {out_path}")
    sys.exit(0)

phase("still", 15)
phase("walk", 35, (CTRL, W), reverse_each=8)
phase("still2", 5)
phase("jog", 35, (W,), reverse_each=8)
phase("still3", 5)
phase("sprint", 35, (SHIFT, W), reverse_each=8)
phase("still4", 5)
seg = {"name": "crouch_down", "start_utc": utc()}
key(C, True); time.sleep(0.1); key(C, False)
wait(2)
seg["end_utc"] = utc(); log["phases"].append(seg)
phase("crouch_walk", 30, (W,), reverse_each=8)
key(C, True); time.sleep(0.1); key(C, False)
wait(2)
seg = {"name": "prone_down", "start_utc": utc()}
key(X, True); time.sleep(0.1); key(X, False)
wait(3)
seg["end_utc"] = utc(); log["phases"].append(seg)
phase("crawl", 25, (W,), reverse_each=8)
key(X, True); time.sleep(0.1); key(X, False)
wait(3)
seg = {"name": "jump", "start_utc": utc()}
for _ in range(6):
    key(V, True); time.sleep(0.1); key(V, False)
    wait(2.5)
seg["end_utc"] = utc(); log["phases"].append(seg)
phase("still5", 5)
# Raise the rifle (hold right), fire single presses.
seg = {"name": "fire", "start_utc": utc()}
button(RIGHT, True)
wait(1.5)
for _ in range(20):
    button(LEFT, True); time.sleep(0.05); button(LEFT, False)
    wait(1.5)
button(RIGHT, False)
seg["end_utc"] = utc(); log["phases"].append(seg)
phase("still6", 10)
let_go()
json.dump(log, open(out_path, "w"), indent=1)
print(f"done: {len(log['phases'])} phases -> {out_path}")
