import time
import numpy as np
import openvino as ov
from openvino import opset10 as opset

core = ov.Core()

def matmul_model(M, K, N, dtype=np.float16):
    P = opset.parameter([M, K], dtype, name="A")
    B = opset.constant(np.random.randn(K, N).astype(dtype) * 0.05, dtype)
    C = opset.matmul(P, B, False, False)
    return ov.Model([C], [P], "mm")

def bench(device, model, M, K, N, warm=3, reps=10):
    ie = core.compile_model(model, device)
    req = ie.create_infer_request()
    A = np.random.randn(M, K).astype(np.float16) * 0.1
    for _ in range(warm):
        req.infer([A])
    best = 1e9
    for _ in range(reps):
        t0 = time.time(); req.infer([A]); best = min(best, time.time() - t0)
    gflops = 2 * M * K * N / 1e9
    gbs = K * N * 2 / 1e9 / best
    return best, gflops / best, gbs

print("== GEMM (prompt-like, f16, oneDNN CPU) ==")
for (M, K, N) in [(4096, 5120, 6144), (1024, 5120, 6144)]:
    m = matmul_model(M, K, N)
    t, tf, gbs = bench("CPU", m, M, K, N)
    print(f"CPU  M{M} K{K} N{N}: {t*1000:8.1f} ms -> {tf:8.1f} GFLOP/s")

print("== GEMV (decode-like, batch 1, f16) ==")
for (K, N) in [(5120, 12288), (5120, 32768)]:
    m = matmul_model(1, K, N)
    t, tf, gbs = bench("CPU", m, 1, K, N, warm=3, reps=20)
    print(f"CPU  K{K} N{N}: {t*1000:8.1f} ms -> {gbs:7.2f} GB/s weights -> {1/t:6.1f} such-GEMV/s")
