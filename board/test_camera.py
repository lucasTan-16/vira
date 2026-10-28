"""板端摄像头采集测试：OpenCV 打开 /dev/video0，读几帧确认正常。
设置 MJPG 640x480@30fps。运行：python3 test_camera.py
"""
import cv2

# 打开摄像头（0 = /dev/video0）
cap = cv2.VideoCapture(0)
# 指定 MJPG（YUYV 高分辨率帧率低）
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("opened:", cap.isOpened())
# 读 5 帧（前几帧可能曝光未稳定）
for i in range(5):
    ok, frame = cap.read()
    if ok:
        print("frame %d: shape %s" % (i, frame.shape))
cap.release()
