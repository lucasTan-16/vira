"""
工具注册表与工具实现。

作用：定义 Agent 能调用的工具（检测、状态查询、设备控制），
     提供工具 JSON（给 LLM）和按名执行的统一入口。
工具真正"做事"的函数在后面注册进来。
"""
import json

# 工具定义：每个工具用 JSON schema 描述名称、功能、参数（给 LLM 看）
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "run_detection",
            "description": "运行一次现场视觉检测，返回人员、PPE 穿戴和违规情况。",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["camera", "image"],
                        "description": "检测来源：指定图像 image 或摄像头实时 camera，默认 image。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_state",
            "description": "查询最近一次检测的现场合规状态，不重新检测。",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "enum": ["all", "violations", "people"],
                        "description": "查询内容：全部/违规/人员，默认 all。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_device",
            "description": "控制报警/停机设备（继电器、LED）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "device": {
                        "type": "string",
                        "enum": ["alarm", "relay", "led"],
                        "description": "要控制的设备。",
                    },
                    "state": {
                        "type": "string",
                        "enum": ["on", "off"],
                        "description": "打开或关闭。",
                    },
                },
                "required": ["device", "state"],
            },
        },
    },
]


class ToolRegistry:
    """工具注册表：管理工具定义和执行函数。"""

    def __init__(self):
        self._funcs = {}   # 工具名 -> 执行函数

    def register(self, name, func):
        """
        注册工具。
        输入：name（工具名，须与 TOOL_DEFS 一致）、func（执行函数）
        """
        self._funcs[name] = func

    def tools_json(self):
        """输出：给 rkllm_set_function_tools 的工具 JSON 字符串。"""
        return json.dumps(TOOL_DEFS, ensure_ascii=False)

    def execute(self, name, arguments):
        """
        按名执行工具。
        输入：name（工具名）、arguments（dict，LLM 给的参数）
        输出：工具执行结果（dict）；未知工具或异常时返回带 error 的 dict
        """
        if name not in self._funcs:
            return {"error": f"unknown tool: {name}"}
        try:
            return self._funcs[name](**arguments)
        except Exception as e:
            return {"error": f"{name} failed: {e}"}

# ===== 工具实现 =====
import os
import sys
import numpy as np

# 让 agent 能 import 板端 board / compliance / capturekit 模块

HERE = os.path.dirname(os.path.abspath(__file__))   # agent/
ROOT = os.path.dirname(HERE)                       # agent/ 的上一级 = 工程根
sys.path.insert(0, os.path.join(ROOT, "board"))
sys.path.insert(0, os.path.join(ROOT, "compliance"))
sys.path.insert(0, ROOT)

from postprocess import decode_three_head, nms
from ppe_rules import evaluate as ppe_evaluate
from capturekit import (                            # 采集统一走抽象层（不再直接用 cv2）
    CameraConfig,
    CaptureConfig,
    CaptureError,
    FrameReadError,
    grab_frame,
)

# 视觉模型与样本图路径
MODEL = os.path.join(ROOT, "models", "best_rknnopt_int8.rknn")
SAMPLE = os.path.join(ROOT, "board", "sample_640.npy")

# 摄像头采集参数：分辨率/像素格式/缓冲由抽象层统一下发，工具层不碰硬件细节
CAMERA_CONFIG = CaptureConfig(
    kind="camera",
    camera=CameraConfig(index=0, width=640, height=480, fourcc="MJPG"),
)


class VisionBackend:
    """视觉后端：rknn 只加载一次，反复检测；NPU 单核 CORE_0。"""

    def __init__(self, model_path=MODEL):
        from rknnlite.api import RKNNLite   # 延迟导入：板端检测时才需要
        self.rk = RKNNLite()
        if self.rk.load_rknn(model_path) != 0:
            raise RuntimeError("load rknn failed")
        self.rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

    def detect(self, frame):
        """
        输入：frame，NHWC uint8 (640,640,3)
        输出：detections (M,6)，每行 [x1,y1,x2,y2,conf,cls]
        """
        out = self.rk.inference(inputs=[frame[None, ...]], data_format="nhwc")
        pred = decode_three_head(out)
        return nms(pred, conf_thres=0.25, iou_thres=0.7)

    def release(self):
        """释放 NPU 资源。"""
        self.rk.release()


# 共享状态：保存最近一次检测结果，供 query_state 读取
STATE = {"last": None}
def _make_summary(results, violations):
    """确定性生成自然语言结论，避免小模型误读数字。"""
    n = len(results)
    v = len(violations)
    if v == 0:
        return f"现场共 {n} 人，全部穿戴合规，没有违规。"
    detail = "；".join(
        f"人员 {p} 缺失 {', '.join(m)}" for p, m in violations)
    return f"现场共 {n} 人，其中 {v} 人违规。{detail}。"

def _capture_frame(index=0):
    """从 USB 摄像头采一帧，返回预处理后的 RGB uint8 (640, 640, 3)。

    采集链路（打开设备 → 下发 MJPG/分辨率 → 读帧 → letterbox → BGR→RGB）
    全部由 capturekit 抽象层实现；本函数只负责把采集层异常翻译成工具层语义。
    """
    config = CAMERA_CONFIG
    if index != 0:
        config = CaptureConfig(
            kind="camera",
            camera=CameraConfig(index=index, width=640, height=480, fourcc="MJPG"),
        )
    try:
        frame = grab_frame(config)
    except FrameReadError as exc:
        raise RuntimeError("camera read failed: %s" % exc) from exc
    except CaptureError as exc:
        raise RuntimeError("camera unavailable: %s" % exc) from exc
    except RuntimeError:
        raise
    return frame.image

def run_detection(source="image"):
    """
    运行一次检测 + 合规判定，并更新共享状态。
    输入：source（image=样本图，camera=摄像头，暂先支持 image）
    输出：人数、检测数、每人缺失装备、违规清单
    """
    if not hasattr(run_detection, "_backend"):
        run_detection._backend = VisionBackend()   # 模型只加载一次
    if source == "camera":
        frame = _capture_frame()
    else:
        frame = np.load(SAMPLE)
    det = run_detection._backend.detect(frame)
    results, violations = ppe_evaluate(det)
    STATE["last"] = {"results": results, "violations": violations}
    return {
        "num_people": len(results),
        "num_violations": len(violations),
        "all_compliant": len(violations) == 0,
        "summary": _make_summary(results, violations),
        "people": [
            {"id": r["person"], "missing_equipment": r["missing"]}
            for r in results
        ],
        "violations": [
            {"person": p, "missing_equipment": m}
            for p, m in violations
        ],
    }



def query_state(item="all"):
    """
    查询最近一次检测的合规状态，不重新检测。
    输入：item（all=全部，violations=只看违规，people=只看人数）
    输出：对应状态；还没检测过则提示先检测
    """
    if STATE["last"] is None:
        return {"message": "no detection yet, call run_detection first"}
    results = STATE["last"]["results"]
    violations = STATE["last"]["violations"]
    if item == "violations":
        return {
            "num_violations": len(violations),
            "summary": _make_summary(results, violations),
            "violations": [
                {"person": p, "missing_equipment": m}
                for p, m in violations
            ],
        }
    if item == "people":
        return {"num_people": len(results)}
    return {
        "num_people": len(results),
        "num_violations": len(violations),
        "all_compliant": len(violations) == 0,
        "summary": _make_summary(results, violations),
        "people": [
            {"id": r["person"], "missing_equipment": r["missing"]}
            for r in results
        ],
        "violations": [
            {"person": p, "missing_equipment": m}
            for p, m in violations
        ],
    }



def control_device(device, state):
    """
    控制报警/停机设备（后续用 libgpiod 接真实 GPIO，这里先模拟）。
    输入：device（alarm/relay/led）、state（on/off）
    输出：已执行的动作记录
    """
    return {"device": device, "state": state, "status": "simulated"}


def build_registry():
    """创建注册表并注册 3 个工具。输出：配置好的 ToolRegistry。"""
    reg = ToolRegistry()
    reg.register("run_detection", run_detection)
    reg.register("query_state", query_state)
    reg.register("control_device", control_device)
    return reg
