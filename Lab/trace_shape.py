"""True tensor shapes, from ONNX Runtime, for a tensor and what feeds it, with the graph above it. Usage: python -I trace_shape.py <model.onnx> <tensor> [depth]"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
path, T = sys.argv[1], sys.argv[2]
DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 5
m = onnx.load(path)
g = m.graph
producer = {o: n for n in g.node for o in n.output}
inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
want, lines = [], []


def walk(name, depth, seen):
    if name in inits:
        v = inits[name]
        lines.append(("  " * depth, name, f"initializer {list(v.shape)} {np.ravel(v)[:5].tolist()}", None))
        return
    n = producer.get(name)
    if n is None:
        lines.append(("  " * depth, name, "graph input", None))
        return
    c = ""
    if n.op_type == "Constant":
        c = " = " + str(np.ravel(numpy_helper.to_array(n.attribute[0].t))[:6].tolist())
    lines.append(("  " * depth, name, n.op_type + c, name))
    want.append(name)
    if depth >= DEPTH or name in seen:
        return
    seen.add(name)
    for i in n.input:
        if i:
            walk(i, depth + 1, seen)


walk(T, 0, set())
names = list(dict.fromkeys(want))
del g.output[:]
for w in names:
    g.output.append(helper.make_empty_tensor_value_info(w))
del g.value_info[:]
s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
vals = dict(zip(names, s.run(None, {g.input[0].name: np.zeros((1, 3, 640, 640), dtype=np.float32)})))
for pad, name, what, key in lines:
    tail = ""
    if key:
        a = np.asarray(vals[key])
        tail = f" -> shape {list(a.shape)}" + (f" = {a.ravel().tolist()}" if a.size <= 6 else "")
    print(f"{pad}{name.replace('/model/decoder/layers.1/self_attn/', '.../')}: {what}{tail}")
