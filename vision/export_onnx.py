"""导出6类 PPE 检测器的固定输入 NCHW ONNX.

输入固定 [1, 3, 640, 640], 不合并NMS（后处理在板端 host 做），
opset=12, simplify、FP32, gong RKMM-Toolkit2 转 FP16/INT18。
"""
from ultralytics import YOLO

# 6 类最优权重
MODEL_PATH = "/home/lucas/vira/vision/runs/ppe6_yolo11n/weights/best.pt"

'''
nms 不写（默认不加）：检测头原始输出[1, 10, 8400]（4坐标+6类），NMS 用 C++ 在板端做
'''
def main():
    model = YOLO(MODEL_PATH)
    out = model.export(
        format="onnx",  # 导出格式
        imgsz=640,      # 与训练一致，改了精度会变化
        opset=12,       # rknn_model_zoo 官方 yolo11 的 算子集版本号(Operator Set Version)
        simplify=True,  # 简化算子图，利于 RKNN 转换
        dynamic=False,  # RKNN 要求固定输入尺寸，动态 batch/ shape 转不了
        half=False,     # ONNX 保持 FP32作为基准，FP16/INT8 量化在RNKK阶段做，才能对比三档
    )
    print("ONNX exported to:", out)


if __name__ == "__main__":
    main()
