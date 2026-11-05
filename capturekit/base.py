"""采集抽象层：FrameSource 统一契约。

设计目标：
1. **上层零感知硬件差异**——推理/调度侧只依赖 ``FrameSource.read()``，
   不出现 ``cv2.VideoCapture`` 之类的硬件调用；
2. **生命周期显式**——open/close 成对，支持 with 上下文，close 幂等；
3. **故障语义清晰**——设备不可用抛 :class:`DeviceUnavailableError`（不可自愈），
   单帧失败抛 :class:`FrameReadError`（可重试），EOF 返回 ``None``；
4. **可测**——不依赖真实摄像头的源（合成/视频/图像）可在 CI 环境验证全链路。
"""
from __future__ import annotations

import abc
import logging
import time
from typing import Iterator

import numpy as np

from .errors import DeviceUnavailableError, FrameReadError, SourceClosedError
from .models import Frame, SourceStats
from .preprocess import Preprocessor

logger = logging.getLogger(__name__)


class FrameSource(abc.ABC):
    """采集源抽象基类。

    子类只需实现 ``_open_impl`` / ``_read_impl`` / ``_close_impl`` 三个钩子，
    序号、时间戳、预处理、统计、异常分类均由基类统一处理。

    Attributes:
        kind: 源类型标识（``camera`` / ``video`` / ``image`` / ``synthetic``）。
        name: 实例标识，用于日志与下游统计（如 ``camera:0``）。
    """

    kind: str = "base"

    def __init__(
        self,
        name: str | None = None,
        preprocessor: Preprocessor | None = None,
        max_read_failures: int = 5,
    ) -> None:
        """
        Args:
            name: 实例标识；缺省用 ``kind``。
            preprocessor: 预处理对象；缺省 letterbox 640x640 + RGB。
            max_read_failures: 连续读失败上限；达到后视为设备故障，抛
                :class:`DeviceUnavailableError`，不再无限重试。
        """
        if max_read_failures < 1:
            raise ValueError("max_read_failures must be >= 1")
        self.name = name or self.kind
        self.stats = SourceStats()
        self.max_read_failures = max_read_failures
        self._preprocessor = preprocessor or Preprocessor()
        self._opened = False
        self._seq = 0
        self._consecutive_failures = 0

    # ===== 只读属性 =====
    @property
    def preprocessor(self) -> Preprocessor:
        """当前生效的预处理对象（读取配置后的实际值，便于日志与核对）。"""
        return self._preprocessor

    @property
    def is_opened(self) -> bool:
        return self._opened

    # ===== 生命周期 =====
    def open(self) -> "FrameSource":
        """打开源。重复调用幂等。"""
        if self._opened:
            return self
        self._open_impl()
        self._opened = True
        self._seq = 0
        self._consecutive_failures = 0
        self.stats = SourceStats(opened_at=time.perf_counter())
        logger.info("source opened: %s", self.name)
        return self

    def close(self) -> None:
        """关闭源并释放硬件/文件句柄。重复调用幂等。"""
        if not self._opened:
            return
        try:
            self._close_impl()
        finally:
            self._opened = False
            self.stats.closed_at = time.perf_counter()
            logger.info("source closed: %s (%s)", self.name, self.stats.as_dict())

    @property
    def is_open(self) -> bool:
        """源是否处于打开状态。"""
        return self._opened

    def __enter__(self) -> "FrameSource":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ===== 读取 =====
    def read(self) -> Frame | None:
        """读取下一帧。

        Returns:
            预处理后的 :class:`Frame`；数据流结束（视频读完、源已排空）返回 ``None``。

        Raises:
            SourceClosedError: 未打开或已关闭。
            FrameReadError: 单帧读取失败，未达重试上限。
            DeviceUnavailableError: 连续失败超过上限，设备判定为不可用。
        """
        if not self._opened:
            raise SourceClosedError(f"source not open: {self.name}")

        raw = self._read_impl()
        if raw is None:
            return None

        self._consecutive_failures = 0
        image = self._preprocessor(raw)
        frame = Frame(
            seq=self._seq,
            image=image,
            ts_capture=time.perf_counter(),
            source=self.name,
            meta={"raw_shape": tuple(raw.shape)},
        )
        self._seq += 1
        self.stats.frames_read += 1
        return frame

    def __iter__(self) -> Iterator[Frame]:
        return self

    def __next__(self) -> Frame:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame

    # ===== 子类钩子 =====
    @abc.abstractmethod
    def _open_impl(self) -> None:
        """打开底层设备/文件；失败抛 :class:`DeviceUnavailableError`。"""

    @abc.abstractmethod
    def _read_impl(self) -> np.ndarray | None:
        """读一帧原始 BGR 图像；结束返回 ``None``；失败抛 :class:`FrameReadError`。"""

    def _close_impl(self) -> None:
        """释放底层资源；默认无操作。"""

    # ===== 内部工具 =====
    def _record_failure(self, reason: str) -> None:
        """累计一次读失败；超过上限升级为设备不可用。"""
        self._consecutive_failures += 1
        self.stats.read_errors += 1
        if self._consecutive_failures >= self.max_read_failures:
            raise DeviceUnavailableError(
                f"{self.name}: {self._consecutive_failures} consecutive read failures, "
                f"last reason: {reason}"
            ) from None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name} open={self._opened}>"


__all__ = [
    "FrameSource",
    "DeviceUnavailableError",
    "FrameReadError",
    "SourceClosedError",
]
