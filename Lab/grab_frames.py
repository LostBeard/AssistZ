"""
Grabs the DayZ client window to JPEGs for the Vision Module's labelled set: each file is named with its UTC milliseconds, so the pose log gives its true position and heading. Asks the window itself to draw (PrintWindow with PW_RENDERFULLCONTENT), so whatever is on top of it does not matter: a first version read the screen and saved the operator's terminal for a whole test run. Nothing is raised or made topmost. Local files on the work disk; the game's images are never committed.
TJ: the game does not render well without focus. So each file name ends in _f (the game had focus) or _u (it did not); use _f frames. Focus is taken at the start, and back only after 30 s with no keyboard or mouse input, so this never fights a person using the PC. ASSISTZ_NO_REFOCUS=1 leaves focus alone, for tests.
A frame that is blank, or that does not change between views, is reported at the end: check.txt in the output folder holds the counts.
Usage: python grab_frames.py <out dir> <seconds> [frames per second]
"""
import ctypes, datetime, json, os, sys, time
from ctypes import wintypes
import numpy as np
from PIL import Image

out, seconds = sys.argv[1], float(sys.argv[2])
fps = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
os.makedirs(out, exist_ok=True)
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32
user32.SetProcessDPIAware()
for f, res, args in ((user32.GetDC, wintypes.HDC, [wintypes.HWND]), (user32.ReleaseDC, ctypes.c_int, [wintypes.HWND, wintypes.HDC]), (gdi32.CreateCompatibleDC, wintypes.HDC, [wintypes.HDC]),
                     (gdi32.CreateCompatibleBitmap, wintypes.HBITMAP, [wintypes.HDC, ctypes.c_int, ctypes.c_int]), (gdi32.SelectObject, wintypes.HGDIOBJ, [wintypes.HDC, wintypes.HGDIOBJ]),
                     (gdi32.DeleteObject, wintypes.BOOL, [wintypes.HGDIOBJ]), (gdi32.DeleteDC, wintypes.BOOL, [wintypes.HDC]), (user32.PrintWindow, wintypes.BOOL, [wintypes.HWND, wintypes.HDC, wintypes.UINT]),
                     (gdi32.GetDIBits, ctypes.c_int, [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT, ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT])):
    f.restype = res
    f.argtypes = args


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long), ("biHeight", ctypes.c_long), ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", ctypes.c_long), ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]


def windows():
    found = []
    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        h = kernel32.OpenProcess(0x1000, False, pid.value)
        name = ctypes.create_unicode_buffer(260)
        size = wintypes.DWORD(260)
        if h and kernel32.QueryFullProcessImageNameW(h, 0, name, ctypes.byref(size)) and name.value.lower().endswith("dayzdiag_x64.exe"):
            r = wintypes.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(r))
            t = ctypes.create_unicode_buffer(200)
            user32.GetWindowTextW(hwnd, t, 200)
            found.append((r.right * r.bottom, hwnd, r.right, r.bottom, t.value, pid.value))
        if h:
            kernel32.CloseHandle(h)
        return True
    user32.EnumWindows(ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(cb), 0)
    return sorted(found, reverse=True)


win = windows()
print("DayZ windows:", [(w[4], w[2], w[3], w[5]) for w in win], flush=True)
assert win, "no DayZ window"
area, hwnd, width, height, title, pid = win[0]
assert width >= 640, f"largest DayZ window is {width} wide: not the game"
json.dump(dict(title=title, width=width, height=height, pid=pid, method="PrintWindow PW_RENDERFULLCONTENT"), open(os.path.join(out, "window.json"), "w"))


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]


user32.GetForegroundWindow.restype = wintypes.HWND
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
no_refocus = os.environ.get("ASSISTZ_NO_REFOCUS") == "1"


def idle_seconds():
    info = LASTINPUTINFO(ctypes.sizeof(LASTINPUTINFO), 0)
    user32.GetLastInputInfo(ctypes.byref(info))
    return (kernel32.GetTickCount() - info.dwTime) / 1000.0


def take_focus():
    # An ALT tap lets a background process set the foreground window.
    user32.keybd_event(0x12, 0, 0, 0)
    user32.keybd_event(0x12, 0, 2, 0)
    user32.ShowWindow(hwnd, 5)
    return user32.SetForegroundWindow(hwnd)


def has_focus():
    return user32.GetForegroundWindow() == hwnd


def grab():
    """The window's client area as an RGB array, or None."""
    r = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(r)) or r.right == 0:
        return None
    w, h = r.right, r.bottom
    wdc = user32.GetDC(hwnd)
    mdc = gdi32.CreateCompatibleDC(wdc)
    bmp = gdi32.CreateCompatibleBitmap(wdc, w, h)
    old = gdi32.SelectObject(mdc, bmp)
    PW_CLIENTONLY, PW_RENDERFULLCONTENT = 1, 2
    ok = user32.PrintWindow(hwnd, mdc, PW_CLIENTONLY | PW_RENDERFULLCONTENT)
    info = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
    buf = (ctypes.c_ubyte * (w * h * 4))()
    lines = gdi32.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(info), 0)
    gdi32.SelectObject(mdc, old)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mdc)
    user32.ReleaseDC(hwnd, wdc)
    if not ok or lines != h:
        return None
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, [2, 1, 0]].copy()


take_focus()
time.sleep(1.0)
print("focus at start:", has_focus(), flush=True)
t0 = time.time()
n = blank = failed = same = unfocused = refocus = run_of_failures = 0
last = None
try:
    while time.time() - t0 < seconds:
        t = time.time()
        focused = has_focus()
        if not focused and not no_refocus and idle_seconds() > 30:
            take_focus()
            refocus += 1
        a = grab()
        if a is None:
            failed += 1
            run_of_failures += 1
            # The client has gone (a Steam error ended one survey): stop, so the run ends now and not on its timer.
            if run_of_failures > 20:
                print("window will not draw: stopping", flush=True)
                break
        else:
            run_of_failures = 0
            small = a[::8, ::8].astype(np.float32).mean(axis=2)
            if small.std() < 2.0:
                blank += 1
            if last is not None and np.abs(small - last).mean() < 0.2:
                same += 1
            last = small
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%H%M%S%f")[:9]
            Image.fromarray(a).save(os.path.join(out, f"{stamp}_{'f' if focused else 'u'}.jpg"), quality=88)
            n += 1
            unfocused += not focused
        time.sleep(max(0, 1 / fps - (time.time() - t)))
finally:
    summary = f"{n} frames in {time.time() - t0:.0f} s; without focus {unfocused}, focus retaken {refocus} times; blank {blank}, same as the last {same}, failed {failed}"
    print(summary, flush=True)
    open(os.path.join(out, "check.txt"), "w").write(summary + "\n")
