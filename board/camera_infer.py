"""板端摄像头实时推理：采集 → letterbox → NPU 推理 → 后处理，全链路。
采集线程(OpenCV MJPG) → 推理线程(NPU) → 后处理线程(CPU)，队列连接。
请用 taskset f0 绑大核运行：taskset f0 python3 camera_infer.py int8
"""
import os, sys, time, threading, queue
import cv2
import numpy as np
from rknnlite.api import RKNNLite
from postprocess import decode_three_head, nms

# ========== 1. 路径与参数 ==========
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODE = sys.argv[1] if len(sys.argv) > 1 else "int8"
MODEL = os.path.join(ROOT, "models", f"best_rknnopt_{MODE}.rknn")
N = 300                       # 处理帧数

# ========== 2. letterbox：任意尺寸 → 640x640（居中补灰边） ==========
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (new_shape[0] - nh) // 2, (new_shape[1] - nw) // 2
    canvas = np.full((new_shape[0], new_shape[1], 3), color, np.uint8)
    canvas[top:top + nh, left:left + nw] = img
    return canvas

# ========== 3. 加载模型（单核 CORE_0） ==========
rk = RKNNLite()
rk.load_rknn(MODEL)
rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

# 帧队列：采集→推理、推理→后处理
q_in = queue.Queue(maxsize=4)
q_out = queue.Queue(maxsize=4)

# ========== 4. 采集线程：读帧 + letterbox + BGR→RGB ==========
def capture():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for i in range(N):
        ok, frame = cap.read()
        if not ok:
            continue
        frame = letterbox(frame)                       # → 640x640
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_in.put((i, frame[None, ...]))                # NHWC
    q_in.put(None)                                    # 结束标志
    cap.release()

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

# ========== 7. 启动并计时 ==========
threads = [threading.Thread(target=capture),
           threading.Thread(target=infer_worker),
           threading.Thread(target=post_worker)]
t0 = time.perf_counter()
for t in threads:
    t.start()
threads[2].join()                  # 后处理结束 = 全部处理完
wall = time.perf_counter() - t0
for t in threads[:2]:
    t.join()

print("模式 %s, 摄像头实时处理 %d 帧" % (MODE, N))
print("墙钟: %.3f s" % wall)
print("实时帧率: %.1f FPS (%.1f ms/帧)" % (N / wall, wall / N * 1000))
rk.release()
