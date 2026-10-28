"""三核推理任意 rknn 模型，配合 RKNN_LOG_LEVEL=4 看 WorkLoad。
用法：python3 test_any.py <model.rknn> [size]
"""
import sys
import numpy as np
from rknnlite.api import RKNNLite

model = sys.argv[1]
h = int(sys.argv[2]) if len(sys.argv) > 2 else 224

rk = RKNNLite(verbose=False)
assert rk.load_rknn(model) == 0
rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
x = np.random.randint(0, 255, (1, h, h, 3), np.uint8)
for _ in range(5):
    rk.inference(inputs=[x], data_format="nhwc")
for _ in range(20):
    rk.inference(inputs=[x], data_format="nhwc")
rk.release()
print("done")
