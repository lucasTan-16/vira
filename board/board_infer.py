"""板端 NPU 视觉推理计时：best_rknnopt fp16/int8。
输入 NHWC 0-255 [1,640,640,3]。运行：python3 board_infer.py fp16|int8
"""
import os, sys, time
import numpy as np
from rknnlite.api import RKNNLite

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # board/ 的上一级 = 工程根
MODE = sys.argv[1] if len(sys.argv) > 1 else "fp16"
MODEL = os.path.join(ROOT, "models", f"best_rknnopt_{MODE}.rknn")
N = 200


def bench(core_mask, name):
    rk = RKNNLite()
    assert rk.load_rknn(MODEL) == 0
    rk.init_runtime(core_mask=core_mask)
    x = np.random.randint(0, 255, (1, 640, 640, 3), dtype=np.uint8)

    def run():
        return rk.inference(inputs=[x], data_format="nhwc")

    for _ in range(10):
        run()
    t0 = time.perf_counter()
    for _ in range(N):
        run()
    dt = (time.perf_counter() - t0) / N * 1000
    print("%s: 单帧 NPU = %.3f ms (%.1f FPS), %d 次" % (name, dt, 1000 / dt, N))
    rk.release()


bench(RKNNLite.NPU_CORE_0, "单核 CORE_0")
bench(RKNNLite.NPU_CORE_0_1_2, "三核 AUTO")
