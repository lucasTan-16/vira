"""WSL/PC 专用：取一张 test 图 letterbox 到 640 NHWC，存 sample_640.npy。
板端无 cv2，此脚本只在开发端跑（运行前激活 rknn-env）。
"""
import cv2, glob
import numpy as np

TEST = "/home/lucas/vira/vision/datasets_ppe6/images/test"
OUT = "/home/lucas/vira/board/sample_640.npy"

# 取第一张 test 图（文件名排序）
img_path = sorted(glob.glob(TEST + "/*.jpg"))[0]
img = cv2.imread(img_path)
h, w = img.shape[:2]

# letterbox：等比缩放到 640 内，再居中补灰边
r = min(640 / h, 640 / w)
nw, nh = int(round(w * r)), int(round(h * r))
img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
canvas = np.full((640, 640, 3), 114, np.uint8)
top, left = (640 - nh) // 2, (640 - nw) // 2
canvas[top:top + nh, left:left + nw] = img

# BGR → RGB（模型输入为 RGB），结果 NHWC
rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
np.save(OUT, rgb)
print("saved", OUT, rgb.shape, "from", img_path)
