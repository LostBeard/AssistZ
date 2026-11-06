"""Rebuild the size-baked constants of a YOLOv8n ONNX for another input size. The generator must first reproduce the 640 originals exactly. Lab copy only, never committed. Usage: python -I yolo_resize.py <in.onnx> <out.onnx> <size>"""
import sys
import numpy as np
import onnx
from onnx import numpy_helper


def grids(size):
    xs, ys, st = [], [], []
    for s in (8, 16, 32):
        n = size // s
        y, x = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5, indexing="ij")
        xs.append(x.ravel()); ys.append(y.ravel()); st.append(np.full(n * n, s))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(st)


def const(m, name):
    for n in m.graph.node:
        if n.op_type == "Constant" and n.output[0] == name:
            return n
    raise KeyError(name)


src, out, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = onnx.load(src)
x, y, st = grids(640)
assert x.size == 8400
got = numpy_helper.to_array(const(m, "/model.22/Constant_9_output_0").attribute[0].t)
assert np.array_equal(got[0, 0], x.astype(np.float32)) and np.array_equal(got[0, 1], y.astype(np.float32)), "anchors differ from the generator"
assert np.array_equal(got, numpy_helper.to_array(const(m, "/model.22/Constant_10_output_0").attribute[0].t))
assert np.array_equal(numpy_helper.to_array(const(m, "/model.22/Constant_12_output_0").attribute[0].t)[0], st.astype(np.float32)), "strides differ"
print("generator reproduces the 640 constants exactly")
x, y, st = grids(size)
count = x.size
anchors = np.stack([x, y])[None].astype(np.float32)


def setc(name, arr):
    n = const(m, name)
    n.attribute[0].t.CopyFrom(numpy_helper.from_array(arr, name=n.attribute[0].t.name))


setc("/model.22/Constant_9_output_0", anchors)
setc("/model.22/Constant_10_output_0", anchors)
setc("/model.22/Constant_12_output_0", st[None].astype(np.float32))
setc("/model.22/dfl/Constant_output_0", np.array([1, 4, 16, count], dtype=np.int64))
setc("/model.22/dfl/Constant_1_output_0", np.array([1, 4, count], dtype=np.int64))
d = m.graph.input[0].type.tensor_type.shape.dim
d[2].dim_value = size; d[3].dim_value = size
m.graph.output[0].type.tensor_type.shape.dim[2].dim_value = count
del m.graph.value_info[:]
# Anything else still naming the old count or size?
for n in m.graph.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                t = numpy_helper.to_array(a.t)
                if t.size and t.size < 50 and np.isin(np.ravel(t), [8400, 6400, 1600, 400, 640, 80, 40, 20]).any():
                    print("still baked?", n.output[0], np.ravel(t).tolist())
    if n.op_type == "Resize":
        print("Resize", n.name, list(n.input))
onnx.checker.check_model(m)
onnx.save(m, out)
print("saved", out, "count", count)
