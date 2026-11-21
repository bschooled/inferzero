import numpy as np, openvino as ov
from openvino import opset10 as opset
P = opset.parameter([64, 256], np.float16, name="A")
B = opset.constant(np.random.randn(256, 128).astype(np.float16), np.float16)
C = opset.matmul(P, B, False, False)
m = ov.Model([C], [P], "mm")
core = ov.Core()
for dev in ("CPU", "NPU"):
    try:
        ie = core.compile_model(m, dev)
        print(dev, "OK", ie.output(0).get_partial_shape())
    except Exception as e:
        print(dev, "ERR:", str(e)[:400])
