"""验证 best.onnx 与 best.pt 在固定随机输入下数值一致（算子级，NMS 前）。

固定随机输入 [1,3,640,640]（0-1，seed=0）；比原始输出张量。
FP32 参考：cosine > 0.9999、max abs diff < 1e-3 视为通过。
"""
import numpy as np
import onnx
import torch
import onnxruntime as ort
from ultralytics import YOLO

PT_PATH = "/home/lucas/vira/vision/runs/ppe6_yolo11n/weights/best.pt"
ONNX_PATH = "/home/lucas/vira/vision/runs/ppe6_yolo11n/weights/best.onnx"


def log(msg):
    print(msg, flush=True)


def main():
    gm = onnx.load(ONNX_PATH)
    log("opset: %s ir: %s" % (gm.opset_import[0].version, gm.ir_version))
    for i in gm.graph.input:
        log("input : %s %s" % (i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]))
    for o in gm.graph.output:
        log("output: %s %s" % (o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]))

    rng = np.random.default_rng(0)
    x = rng.random((1, 3, 640, 640), dtype=np.float32)

    log("loading torch model...")
    model = YOLO(PT_PATH, task="detect").model
    model.fuse()
    model.eval().float()
    # 切到导出模式：8.3.0 需同时设置 Detect.export 和 Detect.format
    # （format 本由官方 exporter 统一设置，手动补；format=onnx 走非 TF 分支）
    for mm in model.modules():
        if mm.__class__.__name__ == "Detect":
            mm.export = True
            mm.format = "onnx"
    log("export mode: Detect.export=True, format=onnx")


    log("torch forward...")
    with torch.no_grad():
        ypt = model(torch.from_numpy(x))
    if isinstance(ypt, (list, tuple)):
        log("torch output is list/tuple len=%d, taking [0]" % len(ypt))
        ypt = ypt[0]
    ypt = np.asarray(ypt.cpu().numpy(), dtype=np.float32)
    log("torch out shape: %s" % (ypt.shape,))

    log("onnxruntime session...")
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    log("ort run...")
    yox = sess.run(None, {inp.name: x})[0].astype(np.float32)
    log("ort out shape: %s" % (yox.shape,))

    ypt = ypt.reshape(yox.shape)
    d = np.abs(ypt - yox)
    a, b = ypt.ravel(), yox.ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    log("max abs diff : %s" % float(d.max()))
    log("mean abs diff: %s" % float(d.mean()))
    log("cosine       : %s" % cos)


if __name__ == "__main__":
    main()
