"""
Is there a sound only close passes have? Every clean shot (no other report near it) with a logged impact is lined up on its fire
time, taken from the report itself (report onset - d / 330, DayZ's measured sound delay), so clock errors drop out. Its real pass
distance is the closest approach of the muzzle -> impact segment to the listener's head; the bullet reaches that point d / 785 s after
the fire (7.62x54). Band levels over the shot's own background are compared between pass-distance groups, in time from the fire and
in time from the bullet's arrival. Nothing here assumes what the sound is.
Usage: python analyze_flyby.py <run dir> [...]
"""
import csv, json, os, sys
import numpy as np
from scipy.stats import mannwhitneyu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_clock import calibrated_reports, envelope_db, HOP

BANDS = ((300, 1000), (1000, 3000), (3000, 6000), (6000, 12000), (12000, 20000))
GROUPS = ((0, 5), (5, 15), (15, 35))
V = 785.0
RATE = 48000
DT = HOP / RATE
FRAME = 480


def band_levels(pcm):
    """dB per band per 5 ms hop (10 ms Hann frames)."""
    mono = pcm.mean(axis=1)
    n = (len(mono) - FRAME) // HOP
    window = np.hanning(FRAME)
    freqs = np.fft.rfftfreq(FRAME, 1 / RATE)
    out = np.zeros((n, len(BANDS)))
    for start in range(0, n, 4096):
        idx = np.arange(start, min(start + 4096, n))
        frames = np.stack([mono[i * HOP: i * HOP + FRAME] * window for i in idx])
        spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        for b, (lo, hi) in enumerate(BANDS):
            out[idx, b] = 10 * np.log10(spec[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) / FRAME + 1e-12)
    return out


def shots_of(run):
    if not os.path.exists(run + "/impacts_client.csv"):
        return []
    pcm = np.load(run + "/audio.npy").astype(np.float64)
    env = envelope_db(pcm)
    levels = band_levels(pcm)
    reports, shift = calibrated_reports(run)
    client = list(csv.DictReader(open(run + "/aishots_client.csv")))
    impacts = list(csv.DictReader(open(run + "/impacts_client.csv")))
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    owner = {}
    for x in impacts:
        t = int(x["game_ms"])
        before = [j for j, c in enumerate(client) if int(c["game_ms"]) <= t <= int(c["game_ms"]) + 2000]
        if before:
            owner.setdefault(max(before, key=lambda j: int(client[j]["game_ms"])), x)
    head = np.array([float(pose[-1]["x"]), float(pose[-1]["y"]) + 1.6, float(pose[-1]["z"])])
    dues = [r[0] for r in reports]
    rows = []
    for i, r in enumerate(reports):
        x = owner.get(i)
        due, d = r[0], r[3]
        a, b = int((due - 0.06) / DT), int((due + 0.06) / DT)
        if x is None or a < 0 or b >= len(env):
            continue
        peak = a + int(np.argmax(env[a:b]))
        onset = peak
        while onset > peak - int(0.06 / DT) and env[onset - 1] > env[peak] - 6:
            onset -= 1
        fire = onset * DT - d / 330.0
        if any(j != i and fire - 0.3 <= dues[j] <= fire + 0.35 for j in range(len(dues))):
            continue
        m = np.array([float(client[i][k]) for k in ("mx", "my", "mz")])
        q = np.array([float(x[k]) for k in ("x", "y", "z")])
        seg = q - m
        t = np.clip(np.dot(head - m, seg) / np.dot(seg, seg), 0, 1)
        closest = m + t * seg
        along = float(np.linalg.norm(closest - m))
        f0 = int(fire / DT)
        lo, hi = f0 - int(0.3 / DT), f0 + int(0.3 / DT)
        if lo < 0 or hi >= len(levels):
            continue
        base = np.median(levels[f0 - int(0.25 / DT): f0 - int(0.05 / DT)], axis=0)
        rows.append(dict(run=os.path.basename(run.rstrip("/\\")), i=i, d=d, passed=float(np.linalg.norm(closest - head)), arrival=along / V,
                         shape=levels[lo:hi] - base, f0=int(0.3 / DT), landed=float(np.linalg.norm(q - head))))
    print(f"{os.path.basename(run.rstrip('/'))}: clock {shift * 1000:+.0f} ms; {len(reports)} shots, {len(rows)} clean with an impact")
    return rows


rows = []
for path in sys.argv[1:]:
    rows += shots_of(path)
groups = [[r for r in rows if lo <= r["passed"] < hi] for lo, hi in GROUPS]
print("\ngroups:", ", ".join(f"pass {lo}-{hi} m: {len(g)} shots, shooter {np.median([r['d'] for r in g]) if g else 0:.0f} m" for (lo, hi), g in zip(GROUPS, groups)))
print("\nMedian band level over the shot's background (dB), by time from the FIRE (ms). The report is at about d / 330:")
times = list(range(-20, 171, 10))
for b, (lo, hi) in enumerate(BANDS):
    print(f"\n  band {lo}-{hi} Hz")
    print("    " + "pass m".ljust(8) + " ".join(f"{t:5d}" for t in times))
    for (glo, ghi), g in zip(GROUPS, groups):
        if not g:
            continue
        cells = []
        for t in times:
            k = int(round(t / 1000 / DT))
            cells.append(np.median([r["shape"][r["f0"] + k, b] for r in g]))
        print("    " + f"{glo}-{ghi}".ljust(8) + " ".join(f"{c:5.1f}" for c in cells))


def window_peak(r, b, start_ms, end_ms):
    k0 = r["f0"] + int(round((r["arrival"] + start_ms / 1000) / DT))
    k1 = r["f0"] + int(round((r["arrival"] + end_ms / 1000) / DT))
    return float(r["shape"][k0: k1 + 1, b].max())


print("\nPeak over background in a window around the bullet's arrival: close (< 5 m) vs far (15-35 m), per band; one-sided Mann-Whitney")
close, far = groups[0], groups[2]
for start, end in ((-15, 10), (-5, 25), (10, 45)):
    print(f"  arrival {start:+d}..{end:+d} ms")
    for b, (lo, hi) in enumerate(BANDS):
        if len(close) < 3 or len(far) < 3:
            print("    too few shots")
            break
        c = [window_peak(r, b, start, end) for r in close]
        f = [window_peak(r, b, start, end) for r in far]
        p = mannwhitneyu(c, f, alternative="greater").pvalue
        print(f"    {lo:>5}-{hi:<5} Hz: close median {np.median(c):5.1f} dB, far median {np.median(f):5.1f} dB, diff {np.median(c) - np.median(f):+5.1f} dB, p = {p:.4f}")
print("\nClose passes, one by one:")
for r in sorted(close, key=lambda x: x["passed"]):
    print(f"  {r['run'][-15:]} shot {r['i']:2d}: passed {r['passed']:.1f} m, shooter {r['d']:.0f} m, arrival {r['arrival'] * 1000:.0f} ms; arrival -5..+25 ms peak by band: " + " ".join(f"{window_peak(r, b, -5, 25):5.1f}" for b in range(len(BANDS))))


# The 12-20 kHz band was chosen after looking at the data, and a bullet that passes close usually lands close too, so:
HIGH = len(BANDS) - 1
A0, A1 = -15, 10   # ms around the arrival


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


print(f"\n12-20 kHz peak, arrival {A0:+d}..{A1:+d} ms, each shot: pass m, landed m from head, peak dB")
for r in sorted(rows, key=lambda x: x["passed"]):
    r["peak"] = window_peak(r, HIGH, A0, A1)
print("  by pass distance:")
for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 12), (12, 18), (18, 25), (25, 35), (35, 1e9)):
    g = [r["peak"] for r in rows if lo <= r["passed"] < hi]
    if g:
        print(f"    {lo:>3}-{hi if hi < 1e9 else 'inf':<4} m: n={len(g):3d}  median {np.median(g):5.1f}  min {min(g):5.1f}  max {max(g):5.1f}")
from scipy.stats import spearmanr
rho = spearmanr([r["passed"] for r in rows], [r["peak"] for r in rows])
print(f"  Spearman pass distance vs peak: rho {rho.statistic:.3f}, p = {rho.pvalue:.2e}, n = {len(rows)}")
print("\n  impact confound: does the level follow the PASS when the bullet landed far away?")
for name, sel in (("passed < 5 m, landed < 10 m", lambda r: r["passed"] < 5 and r["landed"] < 10),
                  ("passed < 5 m, landed >= 10 m", lambda r: r["passed"] < 5 and r["landed"] >= 10),
                  ("passed 5-15 m, landed >= 15 m", lambda r: 5 <= r["passed"] < 15 and r["landed"] >= 15),
                  ("passed 15-35 m, landed >= 15 m", lambda r: 15 <= r["passed"] < 35 and r["landed"] >= 15)):
    g = [r for r in rows if sel(r)]
    if g:
        print(f"    {name:<34} n={len(g):3d}  peak median {np.median([r['peak'] for r in g]):5.1f}, landed median {np.median([r['landed'] for r in g]):6.1f} m")
for r in sorted(rows, key=lambda x: x["passed"])[:12]:
    print(f"      {r['run'][-15:]} shot {r['i']:2d}: passed {r['passed']:5.1f} m, landed {r['landed']:7.1f} m, peak {r['peak']:5.1f} dB")
print("\n  each run on its own (close = < 5 m, far = >= 15 m):")
for run in sorted({r["run"] for r in rows}):
    c = [r["peak"] for r in rows if r["run"] == run and r["passed"] < 5]
    f = [r["peak"] for r in rows if r["run"] == run and r["passed"] >= 15]
    print(f"    {run[-15:]}: close n={len(c)} median {np.median(c) if c else float('nan'):5.1f}; far n={len(f)} median {np.median(f) if f else float('nan'):5.1f}; AUC {auc(c, f):.3f}")
print("\n  control, same band and length well before the fire (must not separate):")
c = [float(r["shape"][r["f0"] - int(0.2 / DT): r["f0"] - int(0.175 / DT) + 1, HIGH].max()) for r in rows if r["passed"] < 5]
f = [float(r["shape"][r["f0"] - int(0.2 / DT): r["f0"] - int(0.175 / DT) + 1, HIGH].max()) for r in rows if r["passed"] >= 15]
print(f"    close median {np.median(c):5.1f}, far median {np.median(f):5.1f}, AUC {auc(c, f):.3f}")
print("\n  thresholds (candidate rule: peak >= T means a pass under 5 m):")
for T in (12, 15, 18, 20, 22, 25):
    tp = sum(r["peak"] >= T for r in rows if r["passed"] < 5)
    nc = sum(r["passed"] < 5 for r in rows)
    mid = sum(r["peak"] >= T for r in rows if 5 <= r["passed"] < 15)
    nm = sum(5 <= r["passed"] < 15 for r in rows)
    fp = sum(r["peak"] >= T for r in rows if r["passed"] >= 15)
    nf = sum(r["passed"] >= 15 for r in rows)
    print(f"    T={T:2d} dB: close {tp}/{nc}, 5-15 m {mid}/{nm}, far {fp}/{nf}")
print("\n  where the close group's 12-20 kHz energy starts and peaks, ms relative to the bullet's arrival:")
print("    " + " ".join(f"{t:5d}" for t in range(-40, 41, 5)))
print("    " + " ".join(f"{np.median([r['shape'][r['f0'] + int(round((r['arrival'] + t / 1000) / DT)), HIGH] for r in close]):5.1f}" for t in range(-40, 41, 5)))
