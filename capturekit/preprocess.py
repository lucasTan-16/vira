"""帧预处理：letterbox 缩放 + BGR→RGB，输出 NPU 可直接吃的输入。

原实现散落在 board/camera_infer.py、board/pipeline_infer.py 与 agent/tools.py 三处，
此处收敛为单一可测实现，采集层统一调用，避免同一套几何逻辑三份拷贝各自漂移。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .errors import SourceConfigError
from .models import INPUT_SIZE


@dataclass(frozen=True)
class PreprocessConfig:
    """预处理参数。

    Attributes:
        target_size: 目标 (h, w)，默认与模型输入一致。
        pad_value: 补边灰度值；与 YOLO 训练一致为 114。
        to_rgb: 是否 BGR→RGB；OpenCV 读入为 BGR，NPU 输入要求 RGB。
    """

    target_size: tuple[int, int] = INPUT_SIZE
    pad_value: int = 114
    to_rgb: bool = True

    def validate(self) -> None:
        """启动期校验，避免非法尺寸带病跑整条流水线。"""
        if len(tuple(self.target_size)) != 2:
            raise SourceConfigError(f"preprocess.target_size must be (h, w), got {self.target_size!r}")
        if any(int(v) <= 0 for v in self.target_size):
            raise SourceConfigError(f"preprocess.target_size must be positive, got {self.target_size!r}")
        if not 0 <= int(self.pad_value) <= 255:
            raise SourceConfigError(f"preprocess.pad_value must be in [0, 255], got {self.pad_value!r}")


def letterbox(
    image: np.ndarray,
    target_size: tuple[int, int] = INPUT_SIZE,
    pad_value: int = 114,
) -> np.ndarray:
    """等比缩放并居中补边到 ``target_size``，不改变长宽比。

    Args:
        image: HWC BGR/RGB uint8 原始图像。
        target_size: (h, w)。
        pad_value: 补边值。

    Returns:
        形状 ``(h, w, 3)`` 的 uint8 图像。

    Raises:
        ValueError: 图像为空或通道数不为 3。
    """
    if image is None or image.size == 0:
        raise ValueError("letterbox: empty image")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"letterbox: expect HWC 3-channel image, got {image.shape}")

    target_h, target_w = target_size
    h, w = image.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def to_rgb(image: np.ndarray) -> np.ndarray:
    """BGR→RGB。"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prepare_frame(
    image: np.ndarray,
    config: PreprocessConfig = PreprocessConfig(),
) -> np.ndarray:
    """按配置把原始帧处理为模型输入（letterbox → RGB）。"""
    out = letterbox(image, config.target_size, config.pad_value)
    return to_rgb(out) if config.to_rgb else out


class Preprocessor:
    """可复用的预处理对象；无状态，线程安全。"""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return prepare_frame(image, self.config)

    @property
    def target_size(self) -> tuple[int, int]:
        return self.config.target_size
