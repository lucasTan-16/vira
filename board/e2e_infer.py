"""板端端到端帧延迟计时：NPU 推理 + 后处理（decode + NMS）。
优先加载同目录 sample_640.npy（真实图，NHWC uint8），没有则用随机噪声。
单核 CORE_0（一路一核）。运行：python3 e2e_infer.py int8|fp16
"""
import os, sys, time
import numpy as np
from rknnlite.api import RKNNLite

# 复用同目录后处理模块
from postprocess import decode_three_head, nms

# ========== 1. 路径与参数 ==========
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # board/ 的上一级 = 工程根
MODE = sys.argv[1] if len(sys.argv) > 1 else "int8"
MODEL = os.path.join(ROOT, "models", f"best_rknnopt_{MODE}.rknn")
SAMPLE = os.path.join(HERE, "sample_640.npy")
N = 100

# ========== 2. 加载模型（单核，一路一核） ==========
rk = RKNNLite()
assert rk.load_rknn(MODEL) == 0
rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

# ========== 3. 准备输入：优先真实图，否则随机 ==========
if os.path.exists(SAMPLE):
    x = np.load(SAMPLE)[None, ...]           # (640,640,3) → (1,640,640,3)
    print("输入: 真实图 sample_640.npy")
else:
    x = np.random.randint(0, 255, (1, 640, 640, 3), dtype=np.uint8)
    print("输入: 随机噪声（无真实检测，NMS 空跑，后处理偏小）")

# ========== 4. 单帧：NPU + 后处理，分别计时 ==========
npu_s = post_s = 0.0

def one_frame():
    global npu_s, post_s
    t0 = time.perf_counter()
    out = rk.inference(inputs=[x], data_format="nhwc")
    t1 = time.perf_counter()
    pred = decode_three_head(out)
    det = nms(pred, conf_thres=0.25, iou_thres=0.7)
    t2 = time.perf_counter()
    npu_s += t1 - t0
    post_s += t2 - t1
    return det

# ========== 5. 预热 + 正式计时 ==========
for _ in range(10):
    one_frame()
npu_s = post_s = 0.0
det = None
for _ in range(N):
    det = one_frame()

# ========== 6. 输出三段耗时 ==========
npu_ms = npu_s / N * 1000
post_ms = post_s / N * 1000
print("模式 %s, %d 帧（单核 CORE_0），保留检测 %d 个" % (MODE, N, len(det)))
print("NPU 推理:          %.3f ms" % npu_ms)
print("后处理(decode+NMS): %.3f ms" % post_ms)
print("端到端:            %.3f ms (%.1f FPS)" % (
    npu_ms + post_ms, 1000 / (npu_ms + post_ms)))
rk.release()
