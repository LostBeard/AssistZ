"""
A DETR family detector in ONNX Runtime on the CPU, on the same frames, whole frame and tiles as the lab's YOLOv8, in the same results format, so see_score.py can compare them. A reference only: the app runs through SpawnDev.ILGPU.ML. Frames stay local.
Usage: python -I detr_ort.py <model.onnx> <frames dir> <out json> <input px> <person class> <every nth frame> [threads]
"""
import csv, json, os, sys, time
import numpy as np
import onnxruntime as ort
from PIL import Image
model, frames_dir, out_path, size, person, nth = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
threads = int(sys.argv[7]) if len(sys.argv) > 7 else 4
TILES = {"full": None, "tile0": (0, 120, 640, 760), "tile1": (480, 120, 1120, 760), "tile2": (960, 120, 1600, 760)}
so = ort.SessionOptions()
so.intra_op_num_threads = threads
s = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
inp = s.get_inputs()[0].name
names = [o.name for o in s.get_outputs()]
print("inputs", [(i.name, i.shape) for i in s.get_inputs()], "outputs", [(o.name, o.shape) for o in s.get_outputs()], flush=True)
rows = list(csv.DictReader(open(os.path.join(frames_dir, "manifest.csv"), encoding="utf-8")))[::nth]
out, times = [], []
t0 = time.time()
for i, r in enumerate(rows):
    im = Image.open(os.path.join(frames_dir, r["frame"])).convert("RGB")
    for kind, box in TILES.items():
        crop = im if box is None else im.crop(box)
        w, h = crop.size
        ox, oy = (0, 0) if box is None else box[:2]
        a = np.asarray(crop.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
        t = time.time()
        res = dict(zip(names, s.run(None, {inp: a.transpose(2, 0, 1)[None]})))
        times.append(time.time() - t)
        logits, boxes = res["logits"][0], res["pred_boxes"][0]
        scores = 1 / (1 + np.exp(-logits))
        best, top = scores.argmax(axis=1), scores.max(axis=1)
        found = []
        for q in np.nonzero(top >= 0.1)[0]:
            cx, cy, bw, bh = boxes[q] * [w, h, w, h]
            found.append(dict(label="person" if best[q] == person else f"c{best[q]}", conf=float(top[q]), x=float(cx - bw / 2 + ox), y=float(cy - bh / 2 + oy), w=float(bw), h=float(bh)))
        out.append(dict(file=r["frame"], kind=kind, ms=times[-1] * 1000, found=found))
    if i % 100 == 0:
        print(f"{i + 1}/{len(rows)} frames, {time.time() - t0:.0f} s, median {np.median(times) * 1000:.0f} ms an inference", flush=True)
        json.dump(out, open(out_path, "w"))
json.dump(out, open(out_path, "w"))
print(f"done {len(rows)} frames in {time.time() - t0:.0f} s; median {np.median(times) * 1000:.0f} ms an inference on the CPU, {threads} threads", flush=True)
