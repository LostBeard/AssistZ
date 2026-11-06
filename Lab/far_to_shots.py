"""
A lab mode 29 run's shots.csv as shots_ai.csv, the form lab_clock reads; distance is measured from the logged muzzle and listener, not the ladder's nominal step. Only shots marked fired.
Usage: python far_to_shots.py <run>
"""
import csv, math, sys
run = sys.argv[1].replace("\\", "/").rstrip("/")
rows = [r for r in csv.DictReader(open(run + "/shots.csv", encoding="utf-8")) if r["fired"] == "true"]
with open(run + "/shots_ai.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["shot", "game_ms", "distance", "miss", "fired", "shooter", "weapon"])
    for r in rows:
        d = math.dist((float(r["src_x"]), float(r["src_y"]), float(r["src_z"])), (float(r["x"]), float(r["y"]), float(r["z"])))
        w.writerow([r["shot"], r["game_ms"], f"{d:.1f}", r["miss"], "true", "lab", r["weapon"]])
print(f"{len(rows)} shots -> {run}/shots_ai.csv")
