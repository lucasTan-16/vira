"""把 6 类 PPE 的 ONNX 转成 RKNN（FP16 不量化 / INT8 PTQ）。

用法：
    python rknn_convert.py fp     # 输出 out/best_fp16.rknn
    python rknn_convert.py i8     # 输出 out/best_int8.rknn
"""
import sys
from pathlib import Path

from rknn.api import RKNN

HERE = Path(__file__).resolve().parent                       # vision/rknn
ONNX_MODEL = HERE.parent / "runs" / "ppe6_yolo11n" / "weights" / "best_rknnopt.onnx"
CALIB_LIST = HERE / "calib" / "dataset.txt"
OUT_DIR = HERE / "out"
TARGET = "rk3588"

# 喂 0-255 图，内部 (x-0)/255，匹配 ultralytics /255 预处理
MEAN = [[0, 0, 0]]
STD = [[255, 255, 255]]


def convert(do_quant: bool, tag: str, opt_level: int = 3):
    # 默认档位 3 用标准名（与板端/文档一致）；实验档位才加 _oN 后缀，避免覆盖标准产物
    suffix = f"_o{opt_level}" if opt_level != 3 else ""
    out_path = OUT_DIR / f"best_rknnopt_{tag}{suffix}.rknn"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=False)
    print("--> config (optimization_level=%d)" % opt_level)
    rknn.config(mean_values=MEAN, std_values=STD, target_platform=TARGET,
                optimization_level=opt_level)

    print("--> load onnx")
    if rknn.load_onnx(model=str(ONNX_MODEL)) != 0:
        raise RuntimeError("load_onnx failed")

    print("--> build (do_quantization=%s)" % do_quant)
    if do_quant:
        ret = rknn.build(do_quantization=True, dataset=str(CALIB_LIST))
    else:
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise RuntimeError("build failed")

    print("--> export rknn")
    if rknn.export_rknn(str(out_path)) != 0:
        raise RuntimeError("export failed")

    rknn.release()
    print(f"saved: {out_path}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "i8"
    opt_level = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    if mode == "fp":
        convert(False, "fp16", opt_level)
    elif mode == "i8":
        convert(True, "int8", opt_level)
    else:
        raise SystemExit("mode must be 'fp' or 'i8'")



if __name__ == "__main__":
    main()
