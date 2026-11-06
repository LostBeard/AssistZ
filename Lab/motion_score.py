"""From motion_look.py's pairs: how well do simple rules tell moving from still, and forward from back, when the view is not turning? Seen while choosing, so not a test."""
import numpy as np
r = np.load("C:/Users/TJ/AppData/Local/Temp/claude/631e58d4667bd5fc/ae7b0fb7-365f-4d31-b173-80673f47f899/scratchpad/motion.npy")
along, speed, turn, lx, rx, gx, gy, ql, qr, qg = r[:, 1], r[:, 2], r[:, 3], r[:, 4], r[:, 5], r[:, 6], r[:, 7], r[:, 8], r[:, 9], r[:, 10]
flat = np.abs(turn) < 5
moving_truth, still_truth = speed > 0.8, speed < 0.3
for thr in (2, 4):
    said = (np.abs(gy) >= thr) | (np.abs(gx) >= thr) | (np.abs(lx) >= thr) | (np.abs(rx) >= thr)
    a = said[flat & moving_truth].mean() * 100
    b = said[flat & still_truth].mean() * 100
    print(f"any band shifted {thr}+ px: says moving on {a:.0f}% of moving pairs, and on {b:.0f}% of still pairs (n {int((flat & moving_truth).sum())}, {int((flat & still_truth).sum())})")
turning = np.abs(turn) >= 5
said = (np.abs(gy) >= 2)
print(f"while the view turns, ground moves up or down 2+ px on {said[turning & still_truth].mean() * 100:.0f}% of still pairs (n {int((turning & still_truth).sum())}), {said[turning & moving_truth].mean() * 100:.0f}% of moving")
f, b = flat & (along > 0.8), flat & (along < -0.8)
call = np.sign(gy)
print(f"forward or back from the ground's direction, not turning: forward right {100 * (call[f] > 0).mean():.0f}%, wrong {100 * (call[f] < 0).mean():.0f}%, no call {100 * (call[f] == 0).mean():.0f}%; back right {100 * (call[b] < 0).mean():.0f}%, wrong {100 * (call[b] > 0).mean():.0f}%, no call {100 * (call[b] == 0).mean():.0f}%")
s2 = np.array_split(np.arange(len(r)), 3)
for name, lo, hi in (("walk", 0.8, 2), ("jog", 2, 3.2), ("sprint", 3.2, 9)):
    k = flat & (along >= lo) & (along < hi)
    if k.sum():
        print(f"  forward at {name} speed, n {k.sum()}: right {100 * (call[k] > 0).mean():.0f}%, wrong {100 * (call[k] < 0).mean():.0f}%, no call {100 * (call[k] == 0).mean():.0f}%; median y {np.median(gy[k]):.1f}")
for name, lo, hi in (("slow back", -2, -0.8), ("fast back", -9, -2)):
    k = flat & (along >= lo) & (along < hi)
    if k.sum():
        print(f"  {name}, n {k.sum()}: right {100 * (call[k] < 0).mean():.0f}%, wrong {100 * (call[k] > 0).mean():.0f}%, no call {100 * (call[k] == 0).mean():.0f}%")
