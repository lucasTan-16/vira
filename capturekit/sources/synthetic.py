"""合成帧源：不依赖任何硬件/文件，用于 CI、压测与调度策略对比。

用途：
- 摄像头损坏或 CI 环境无设备时，验证采集→调度→推理→后处理全链路；
- 作为"确定性强、速率可控"的产帧器，验证丢弃陈旧帧、丢帧计数与延迟指标；
- 通过 ``frame_factory`` 注入自定义帧内容（如带序号标记的图），验证帧序语义。
"""
from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np

from ..base import FrameSource
from ..errors import FrameReadError
from ..preprocess import Preprocessor

logger = logging.getLogger(__name__)

FrameFactory = Callable[[int, tuple[int, int]], np.ndarray]


def _gradient(seq: int, shape: tuple[int, int]) -> np.ndarray:
    """随 seq 缓慢变化的渐变图，肉眼可辨帧是否在动。"""
    h, w = shape
    base = (seq * 7) % 256
    row = np.arange(w, dtype=np.int32)
    canvas = np.empty((h, w, 3), dtype=np.uint8)
    canvas[..., 0] = (row + base) % 256
    canvas[..., 1] = (row // 2 + base) % 256
    canvas[..., 2] = base
    return canvas


def _noise(seq: int, shape: tuple[int, int]) -> np.ndarray:
    """确定性噪声（种子随 seq 变化），保证可复现。"""
    rng = np.random.default_rng(seed=seq)
    return rng.integers(0, 256, (*shape, 3), dtype=np.uint8)


def _fill(seq: int, shape: tuple[int, int]) -> np.ndarray:
    """纯色图，seq 决定亮度。"""
    h, w = shape
    return np.full((h, w, 3), (seq * 13) % 256, dtype=np.uint8)


_PATTERNS: dict[str, FrameFactory] = {
    "gradient": _gradient,
    "noise": _noise,
    "fill": _fill,
}


class SyntheticSource(FrameSource):
    """合成帧源。

    Args:
        width / height: 原始帧分辨率（预处理前）。
        fps: 产帧节拍；``<=0`` 表示不限速（压测用）。
        total: 总帧数；``0`` 表示无限。
        pattern: ``gradient`` / ``noise`` / ``fill``，或通过 ``frame_factory`` 自定义。
        frame_factory: ``(seq, (h, w)) -> HWC uint8 BGR``，优先级高于 ``pattern``。
        preprocessor: 预处理对象。
    """

    kind = "synthetic"

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        total: int = 0,
        pattern: str = "gradient",
        frame_factory: FrameFactory | None = None,
        preprocessor: Preprocessor | None = None,
        fail_every: int = 0,
    ) -> None:
        if frame_factory is None and pattern not in _PATTERNS:
            raise ValueError(f"unknown pattern: {pattern}; choose from {sorted(_PATTERNS)}")
        self.width = width
        self.height = height
        self.fps = fps
        self.total = total
        self.pattern = pattern
        self.frame_factory = frame_factory
        self.fail_every = fail_every          # >0 时每 N 帧注入一次读失败，验证故障路径
        self._attempt = 0
        super().__init__(name="synthetic", preprocessor=preprocessor)
        self._next_deadline = 0.0

    def _open_impl(self) -> None:
        self._next_deadline = time.perf_counter()
        self._attempt = 0

    def _read_impl(self):
        """按"尝试次数"推进时间轴：失败尝试也消耗一格，避免注入失败后原地死循环。"""
        attempt = self._attempt
        self._attempt += 1
        if self.total and attempt >= self.total:
            return None
        if self.fail_every and attempt and attempt % self.fail_every == 0:
            reason = f"synthetic injected failure at attempt={attempt}"
            self._record_failure(reason)      # 达上限时直接升级为 DeviceUnavailableError
            raise FrameReadError(f"{self.name}: {reason}")
        self._pace()
        factory = self.frame_factory or _PATTERNS[self.pattern]
        return factory(attempt, (self.height, self.width))

    def _pace(self) -> None:
        if self.fps <= 0:
            return
        interval = 1.0 / self.fps
        now = time.perf_counter()
        if now < self._next_deadline:
            time.sleep(self._next_deadline - now)
        self._next_deadline = max(now, self._next_deadline) + interval
