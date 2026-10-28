"""板端单实例峰值内存测量：加载模型 + 推理稳态后，打印进程峰值 RSS（VmHWM）。
单核 CORE_0。运行：python3 measure_mem.py int8|fp16
"""
import os, sys, resource
import numpy as np
from rknnlite.api import RKNNLite

# ========== 1. 路径与参数 ==========
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODE = sys.argv[1] if len(sys.argv) > 1 else "int8"
MODEL = os.path.join(ROOT, "models", f"best_rknnopt_{MODE}.rknn")

# ========== 2. 加载模型 + 跑若干帧，让内存占用达到稳态 ==========
rk = RKNNLite()
rk.load_rknn(MODEL)
rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
x = np.random.randint(0, 255, (1, 640, 640, 3), np.uint8)
for _ in range(20):
    rk.inference(inputs=[x], data_format="nhwc")

# ========== 3. 读取进程峰值 RSS（Linux 上 ru_maxrss 单位 KB） ==========
peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print("模式 %s 单实例峰值 RSS: %.1f MB" % (MODE, peak_kb / 1024))
rk.release()
