"""capturekit —— VIRA 采集抽象层。

职责边界：
- **对外**只暴露 :class:`~capturekit.base.FrameSource`、:class:`~capturekit.models.Frame`
  与帧调度（:class:`~capturekit.scheduler.LatestFrameSlot` /
  :class:`~capturekit.scheduler.FramePump`）；
- **对内**封装设备差异（V4L2 摄像头 / 视频文件 / 静态图 / 合成）、参数下发、
  预处理（letterbox + RGB）、异常分类与统计。

典型用法（最新帧优先，非阻塞）::

    from capturekit import CaptureConfig, CameraConfig, create_source, FramePump, LatestFrameSlot

    cfg = CaptureConfig(kind="camera", camera=CameraConfig(index=0))
    sink = LatestFrameSlot()
    pump = FramePump(create_source(cfg), sink)
    pump.start()
    try:
        frame = sink.get(timeout=1.0)   # 永远拿到最新帧；来不及时旧帧已被丢弃
    finally:
        pump.stop()
        sink.close()

无摄像头环境下用合成源替代（CI / 压测）::

    cfg = CaptureConfig(kind="synthetic")
"""
from __future__ import annotations

from .base import FrameSource
from .config import (
    CameraConfig,
    CaptureConfig,
    ImageConfig,
    SyntheticConfig,
    VideoConfig,
)
from .devices import absence_hint, has_camera, probe_camera_devices
from .errors import (
    CaptureError,
    DeviceUnavailableError,
    FrameReadError,
    SourceClosedError,
    SourceConfigError,
)
from .metrics import CaptureReport, LatencyTracker
from .models import INPUT_SIZE, Frame, SourceStats
from .preprocess import PreprocessConfig, Preprocessor, letterbox, prepare_frame, to_rgb
from .registry import create_source, grab_frame, register_source, supported_kinds
from .scheduler import (
    BlockingFrameQueue,
    FramePump,
    FrameSink,
    LatestFrameSlot,
    Policy,
    SinkStats,
    create_sink,
)
from .sources import CameraSource, ImageSource, SyntheticSource, VideoFileSource

__version__ = "1.0.0"

__all__ = [
    # 抽象与数据模型
    "FrameSource",
    "Frame",
    "SourceStats",
    "INPUT_SIZE",
    # 配置
    "CaptureConfig",
    "CameraConfig",
    "VideoConfig",
    "ImageConfig",
    "SyntheticConfig",
    "PreprocessConfig",
    # 异常
    "CaptureError",
    "SourceConfigError",
    "DeviceUnavailableError",
    "FrameReadError",
    "SourceClosedError",
    # 具体源
    "CameraSource",
    "VideoFileSource",
    "ImageSource",
    "SyntheticSource",
    # 工厂
    "create_source",
    "grab_frame",
    "register_source",
    "supported_kinds",
    # 调度
    "FrameSink",
    "FramePump",
    "LatestFrameSlot",
    "BlockingFrameQueue",
    "SinkStats",
    "Policy",
    "create_sink",
    # 预处理与指标
    "Preprocessor",
    "letterbox",
    "to_rgb",
    "prepare_frame",
    "LatencyTracker",
    "CaptureReport",
    # 设备探测
    "probe_camera_devices",
    "has_camera",
    "absence_hint",
    "__version__",
]
