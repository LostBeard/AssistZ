"""Inputs, outputs and operator types of an ONNX model: does anything look unusual? Usage: python -I onnx_io.py <model.onnx>"""
import collections, sys
import onnx
m = onnx.load(sys.argv[1], load_external_data=False)
g = m.graph
dims = lambda v: [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
print("inputs", [(i.name, dims(i)) for i in g.input])
print("outputs", [(o.name, dims(o)) for o in g.output])
print("nodes", len(g.node), "opset", [o.version for o in m.opset_import])
print("ops", dict(collections.Counter(n.op_type for n in g.node).most_common()))
