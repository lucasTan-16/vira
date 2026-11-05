"""采集源实现集合。

新增源类型时：在此实现 ``FrameSource`` 子类 → 在 :mod:`capturekit.registry` 注册。
"""
from .camera import CameraSource
from .synthetic import SyntheticSource
from .video import ImageSource, VideoFileSource

__all__ = ["CameraSource", "SyntheticSource", "VideoFileSource", "ImageSource"]
