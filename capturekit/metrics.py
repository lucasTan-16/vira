"""采集层指标：延迟滑动窗口 + 聚合报告。

统计口径（任何时刻都应满足）：
    published == consumed + dropped + pending

其中 ``pending`` 为已在槽/队列中尚未被取走的帧数。
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any


class LatencyTracker:
    """固定窗口的延迟统计（线程安全），单位毫秒。

    用定长 deque 保留最近 N 个样本，避免长时间运行内存增长；
    统计值反映"最近一段"的消费延迟，而不是全生命周期均值。
    """

    def __init__(self, window: int = 256) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._samples: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._count = 0
        self._max = 0.0

    def record(self, latency_ms: float) -> None:
        """记录一个延迟样本。"""
        value = max(0.0, float(latency_ms))
        with self._lock:
            self._samples.append(value)
            self._count += 1
            self._max = max(self._max, value)

    @property
    def count(self) -> int:
        """累计样本数（不受窗口长度限制）。"""
        with self._lock:
            return self._count

    @property
    def max_ms(self) -> float:
        """生命周期内最大延迟。"""
        with self._lock:
            return self._max

    def avg_ms(self) -> float:
        with self._lock:
            data = list(self._samples)
        return sum(data) / len(data) if data else 0.0

    def p95_ms(self) -> float:
        with self._lock:
            data = sorted(self._samples)
        if not data:
            return 0.0
        idx = min(len(data) - 1, int(round(0.95 * (len(data) - 1))))
        return data[idx]

    def snapshot(self) -> dict[str, float]:
        return {
            "avg_ms": round(self.avg_ms(), 3),
            "p95_ms": round(self.p95_ms(), 3),
            "max_ms": round(self.max_ms, 3),
            "samples": self.count,
        }


@dataclass(frozen=True)
class CaptureReport:
    """一次采集会话的完整统计快照，用于日志与实验记录。"""

    source: str
    policy: str
    window_s: float
    frames_produced: int
    frames_published: int
    frames_consumed: int
    frames_dropped: int
    read_errors: int
    latency: dict[str, float] = field(default_factory=dict)
    source_detail: dict[str, Any] = field(default_factory=dict)

    @property
    def pending(self) -> int:
        """槽/队列中尚未被消费的帧数。"""
        return self.frames_published - self.frames_consumed - self.frames_dropped

    @property
    def drop_ratio(self) -> float:
        """丢弃率 = dropped / published。"""
        return self.frames_dropped / self.frames_published if self.frames_published else 0.0

    @property
    def producer_fps(self) -> float:
        return self.frames_published / self.window_s if self.window_s > 0 else 0.0

    @property
    def consumer_fps(self) -> float:
        return self.frames_consumed / self.window_s if self.window_s > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "policy": self.policy,
            "window_s": round(self.window_s, 3),
            "frames_produced": self.frames_produced,
            "frames_published": self.frames_published,
            "frames_consumed": self.frames_consumed,
            "frames_dropped": self.frames_dropped,
            "pending": self.pending,
            "drop_ratio": round(self.drop_ratio, 4),
            "read_errors": self.read_errors,
            "producer_fps": round(self.producer_fps, 2),
            "consumer_fps": round(self.consumer_fps, 2),
            "latency": self.latency,
            "source_detail": self.source_detail,
        }

    def format_lines(self) -> list[str]:
        """格式化为可直接打印的多行文本。"""
        lines = [
            f"采集源      : {self.source}",
            f"调度策略    : {self.policy}",
            f"墙钟        : {self.window_s:.3f} s",
            f"产帧/发布   : {self.frames_produced} / {self.frames_published}"
            f"（{self.producer_fps:.1f} FPS）",
            f"消费        : {self.frames_consumed}（{self.consumer_fps:.1f} FPS）",
            f"丢弃陈旧帧  : {self.frames_dropped}（丢弃率 {self.drop_ratio * 100:.1f}%）",
            f"槽内待消费  : {self.pending}",
            f"读失败      : {self.read_errors}",
        ]
        if self.latency:
            lines.append(
                "消费延迟    : avg {avg_ms:.2f} ms / p95 {p95_ms:.2f} ms / max {max_ms:.2f} ms".format(
                    **self.latency
                )
            )
        return lines

    def format_text(self) -> str:
        return "\n".join(self.format_lines())
