import sys
import time
import numpy as np
import openvino as ov
from openvino import opset10 as opset

def main():
    core = ov.Core()
    K, N = 5120, int(sys.argv[3]) if len(sys.argv) > 3 else 16384
    P = opset.parameter([1, K], np.float16, name="A")
    B = opset.constant(np.random.randn(K, N).astype(np.float16) * 0.05, np.float16)
    C = opset.matmul(P, B, False, False)
    m = ov.Model([C], [P], "mm")
    ie = core.compile_model(m, "CPU")
    req = ie.create_infer_request()
    A = np.random.randn(1, K).astype(np.float16) * 0.1
    for _ in range(5):
        req.infer([A])
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
    tag = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    t_end = time.time() + dur
    n = 0
    best = 1e9
    while time.time() < t_end:
        t0 = time.time(); req.infer([A]); dt = time.time() - t0
        best = min(best, dt); n += 1
    gbs = K * N * 2 / 1e9 / best
    print(f"{tag} GEMV K{K} N{N}: {n} runs, best {best*1000:.1f} ms -> {gbs:.1f} GB/s weights", flush=True)

if __name__ == "__main__":
    main()
