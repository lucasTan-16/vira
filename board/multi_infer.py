"""板端 NPU 多实例多线程并行吞吐：3 个实例各绑 1 个 NPU 核。
输入 NHWC 0-255 [1,640,640,3]。
运行：python3 multi_infer.py int8|fp16
目的：验证多实例并行能否在 ConvSwish 单核限制下，把三核吞吐拉到约 3 倍。
"""
import os, sys, time, threading
import numpy as np
from rknnlite.api import RKNNLite

# ========== 1. 路径与参数 ==========
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # board/ 的上一级 = 工程根
MODE = sys.argv[1] if len(sys.argv) > 1 else "int8"          # 选 int8 或 fp16
MODEL = os.path.join(ROOT, "models", f"best_rknnopt_{MODE}.rknn")
N = 100                                                        # 每个线程跑的帧数
# 三个核的掩码：每个实例绑定一个不同的核，互不重叠
CORES = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]

# ========== 2. 线程共享状态 ==========
per_thread_s = [0.0, 0.0, 0.0]   # 记录每个线程正式计时段的总耗时（秒）
# 屏障：3 个线程都到齐（预热完成）后才同时放行，保证三线程同时起跑、计时公平
barrier = threading.Barrier(3)


# ========== 3. 每个线程执行的函数 ==========
def worker(idx, rk):
    """idx=线程/核编号，rk=该线程专属的 RKNNLite 实例。"""
    # 每个线程用独立输入，避免共享 buffer 可能的竞争
    x = np.random.randint(0, 255, (1, 640, 640, 3), dtype=np.uint8)

    # 预热：先跑 10 帧，让 NPU 进入稳定状态（不计入计时）
    for _ in range(10):
        rk.inference(inputs=[x], data_format="nhwc")

    # 在此等待，直到三个线程都预热完，再同时往下走
    barrier.wait()

    # 正式计时段：本线程连续推理 N 帧
    t0 = time.perf_counter()
    for _ in range(N):
        rk.inference(inputs=[x], data_format="nhwc")
    per_thread_s[idx] = time.perf_counter() - t0


# ========== 4. 主线程：加载 3 个实例（不计时） ==========
# 每个实例加载同一份模型，但 init_runtime 时绑定不同的核
rks = []
for core in CORES:
    rk = RKNNLite()
    assert rk.load_rknn(MODEL) == 0          # 加载模型，返回 0 表示成功
    rk.init_runtime(core_mask=core)          # 绑定到指定核
    rks.append(rk)

# ========== 5. 启动 3 个线程并等待全部结束 ==========
threads = []
for i, rk in enumerate(rks):
    t = threading.Thread(target=worker, args=(i, rk))
    t.start()
    threads.append(t)
for t in threads:
    t.join()          # 等所有线程跑完

# ========== 6. 统计并输出 ==========
# 每个线程各自的单帧耗时（验证三核性能是否一致）
for i in range(3):
    ms = per_thread_s[i] / N * 1000
    print("核%d: %.3f ms/帧 (%.1f FPS)" % (i, ms, 1000 / ms))

# 三线程同时起跑，并行墙钟 = 最慢线程的耗时
parallel_s = max(per_thread_s)
# 聚合吞吐 = 三线程总帧数 / 并行墙钟；理想情况约为单实例的 3 倍
print("并行墙钟 %.3f s, 聚合吞吐 %.1f FPS (%d 帧)" % (
    parallel_s, 3 * N / parallel_s, 3 * N))

# ========== 7. 释放三个实例 ==========
for rk in rks:
    rk.release()
