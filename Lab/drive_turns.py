"""
Turns the lab player's view with synthetic RELATIVE mouse moves (Windows SendInput), so a video yaw tracker can be
checked against the pose log with nobody at the keyboard. Safety: only mouse moves are sent (no keys, no clicks), and
before every step the foreground window must belong to the DayZ client PID recorded by launch_dayzdiag.ps1;
otherwise the script stops. Each segment's UTC start/end and mouse delta are written to a JSON log.
Usage: python drive_turns.py <dayzdiag.pids.json> <out.json>
"""
import ctypes, ctypes.wintypes as wt, json, sys, time, datetime

user32 = ctypes.windll.user32
pids = json.load(open(sys.argv[1], encoding="utf-8-sig"))
client_pid = int(pids["client"])
out_path = sys.argv[2]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD), ("dwFlags", wt.DWORD), ("time", wt.DWORD), ("dwExtraInfo", ctypes.c_size_t)]
class INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT), ("pad", ctypes.c_byte * 32)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wt.DWORD), ("u", _U)]
MOUSEEVENTF_MOVE = 0x0001

def move(dx, dy=0):
    inp = INPUT(type=0)
    inp.mi = MOUSEINPUT(dx, dy, 0, MOUSEEVENTF_MOVE, 0, 0)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

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

def foreground_pid():
    p = wt.DWORD()
    user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), ctypes.byref(p))
    return p.value

def focus(hwnd):
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    # Windows only lets the foreground process hand focus over; attach to its input queue to be allowed.
    fg = user32.GetForegroundWindow()
    cur = ctypes.windll.kernel32.GetCurrentThreadId()
    fg_thread = user32.GetWindowThreadProcessId(fg, None)
    user32.AttachThreadInput(cur, fg_thread, True)
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    user32.AttachThreadInput(cur, fg_thread, False)
    time.sleep(0.5)
    return foreground_pid() == client_pid

hwnd = window_of(client_pid)
if not hwnd:
    sys.exit(f"no visible window for DayZ client PID {client_pid}")
if not focus(hwnd):
    sys.exit(f"could not bring DayZ client to the foreground (foreground PID {foreground_pid()})")

# A windowed DayZ only takes mouse look after it has captured the mouse, which a click inside the window does (the
# first automated run showed relative moves alone change nothing). With empty hands the click is a punch: harmless.
MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
rect = wt.RECT()
user32.GetClientRect(hwnd, ctypes.byref(rect))
pt = wt.POINT((rect.right - rect.left) // 2, (rect.bottom - rect.top) // 2)
user32.ClientToScreen(hwnd, ctypes.byref(pt))
user32.SetCursorPos(pt.x, pt.y)
time.sleep(0.2)
if foreground_pid() != client_pid:
    sys.exit("DayZ client lost the foreground before the capture click")
for flag in (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP):
    inp = INPUT(type=0)
    inp.mi = MOUSEINPUT(0, 0, 0, flag, 0, 0)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    time.sleep(0.05)
time.sleep(1.0)

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

# (label, steps, dx per step, step interval s, pause after s)
# "sweeps" (third argument): slow back-and-forth turns, ~+-40 deg over ~3 s each way, for front/back resolution of
# repeating sources (lab mode 3); default: the yaw-tracker calibration plan.
if len(sys.argv) > 3 and sys.argv[3] == "sweeps":
    plan = [("settle", 0, 0, 0, 2.0)]
    for i in range(6):
        plan += [(f"sweep right {i}", 180, 2, 0.016, 0.5), (f"sweep left {i}", 180, -2, 0.016, 0.5)]
else:
  plan = [
    ("settle", 0, 0, 0, 3.0),
    ("right slow", 60, 6, 0.016, 2.0),
    ("left slow", 60, -6, 0.016, 2.0),
    ("right medium", 40, 18, 0.016, 2.0),
    ("left medium", 40, -18, 0.016, 2.0),
    ("right fast", 20, 45, 0.016, 2.0),
    ("left fast", 20, -45, 0.016, 2.0),
    ("right sweep", 150, 8, 0.016, 2.0),
    ("left sweep", 150, -8, 0.016, 3.0),
]
log = {"client_pid": client_pid, "segments": []}
for label, steps, dx, interval, pause in plan:
    seg = {"label": label, "dx_per_step": dx, "steps": steps, "start_utc": utc()}
    for _ in range(steps):
        if foreground_pid() != client_pid:
            seg["aborted"] = "DayZ client lost the foreground"
            log["segments"].append(seg)
            json.dump(log, open(out_path, "w"), indent=1)
            sys.exit("stopped: DayZ client is no longer the foreground window")
        move(dx)
        time.sleep(interval)
    seg["end_utc"] = utc()
    log["segments"].append(seg)
    time.sleep(pause)
json.dump(log, open(out_path, "w"), indent=1)
print(f"done: {len(log['segments'])} segments -> {out_path}")
