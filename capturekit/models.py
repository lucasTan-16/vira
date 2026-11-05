"""采集层数据模型：帧对象与采集统计。

Frame 是采集层与推理层之间的唯一契约：
上游只管产出 Frame，下游只认 Frame，不再关心帧来自摄像头、视频文件还是合成源。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

# 模型输入尺寸（与训练/RKNN 转换一致）
INPUT_SIZE = (640, 640)


@dataclass(frozen=True)
class Frame:
    """一帧已预处理（letterbox + RGB）的图像及其时序元数据。

    Attributes:
        seq: 源内自增序号，从 0 开始；用于判断是否拿到了新帧、统计丢帧跳号。
        image: uint8 HWC RGB，形状为 ``INPUT_SIZE + (3,)``，可直接送 NPU。
        ts_capture: 采集完成时刻（``time.perf_counter()`` 秒），用于算端到端延迟。
        source: 产出该帧的采集源标识，例如 ``camera:0`` / ``synthetic``。
        meta: 附加键值（原始分辨率、是否重连等），便于排查，不参与计算。
    """

    seq: int
    image: np.ndarray
    ts_capture: float
    source: str
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        """图像形状。"""
        return tuple(self.image.shape)

    def age_ms(self, now: float | None = None) -> float:
        """该帧自采集到此刻的年龄（毫秒）；用于陈旧帧判定与延迟统计。"""
        now = time.perf_counter() if now is None else now
        return (now - self.ts_capture) * 1000.0


@dataclass
class SourceStats:
    """单个采集源的累计统计，由 :class:`~capturekit.base.FrameSource` 维护。"""

    frames_read: int = 0
    read_errors: int = 0
    reconnects: int = 0
    opened_at: float | None = None
    closed_at: float | None = None

    @property
    def uptime_s(self) -> float:
        """源从打开到关闭（或至今）的秒数。"""
        if self.opened_at is None:
            return 0.0
        end = self.closed_at if self.closed_at is not None else time.perf_counter()
        return end - self.opened_at

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames_read": self.frames_read,
            "read_errors": self.read_errors,
            "reconnects": self.reconnects,
            "uptime_s": round(self.uptime_s, 3),
        }
