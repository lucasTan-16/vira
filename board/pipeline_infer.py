"""板端连续帧率（流水线）验证：NPU 推理与 CPU 后处理并行。
源线程产帧 → 推理线程(NPU CORE_0) → 后处理线程(decode+NMS)，队列连接。
稳态吞吐 = max(NPU, 后处理)，应明显高于串行端到端。
运行：python3 pipeline_infer.py int8|fp16
"""
import os, sys, time, threading, queue
import numpy as np
from rknnlite.api import RKNNLite
from postprocess import decode_three_head, nms

# ========== 1. 路径与参数 ==========
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODE = sys.argv[1] if len(sys.argv) > 1 else "int8"
MODEL = os.path.join(ROOT, "models", f"best_rknnopt_{MODE}.rknn")
SAMPLE = os.path.join(HERE, "sample_640.npy")
N = 300

# ========== 2. 加载模型（计时前，单核 CORE_0） ==========
rk = RKNNLite()
assert rk.load_rknn(MODEL) == 0
rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
frame = np.load(SAMPLE)[None, ...]

# ========== 3. 预热：让 NPU 和后处理都热起来 ==========
for _ in range(10):
    out = rk.inference(inputs=[frame], data_format="nhwc")
    nms(decode_three_head(out))

# 帧队列：源→推理、推理→后处理（小缓冲，避免积压）
q_in = queue.Queue(maxsize=4)
q_out = queue.Queue(maxsize=4)

# ========== 4. 源线程：产帧（模拟摄像头，真实图重复） ==========
def source():
    for i in range(N):
        q_in.put((i, frame))
    q_in.put(None)                  # 结束标志

# ========== 5. 推理线程：NPU 推理 ==========
def infer_worker():
    while True:
        item = q_in.get()
        if item is None:
            q_out.put(None)
            break
        idx, f = item
        out = rk.inference(inputs=[f], data_format="nhwc")
        q_out.put((idx, out))

# ========== 6. 后处理线程：decode + NMS ==========
def post_worker():
    c = 0
    while True:
        item = q_out.get()
        if item is None:
            return c
        idx, out = item
        nms(decode_three_head(out))
        c += 1

# ========== 7. 启动流水线，测墙钟 ==========
t_source = threading.Thread(target=source)
t_infer = threading.Thread(target=infer_worker)
t_post = threading.Thread(target=post_worker)

t0 = time.perf_counter()
for t in (t_source, t_infer, t_post):
    t.start()
t_post.join()                       # 后处理结束 = 所有帧处理完
wall = time.perf_counter() - t0
t_source.join(); t_infer.join()

print("模式 %s, 流水线 %d 帧" % (MODE, N))
print("并行墙钟: %.3f s" % wall)
print("连续吞吐: %.1f FPS (%.3f ms/帧)" % (N / wall, wall / N * 1000))
rk.release()
