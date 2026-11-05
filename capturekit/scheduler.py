"""帧调度：把"采集→推理"之间的帧传递语义从阻塞积压改为**最新帧优先**。

改造前（D1 问题）：``queue.Queue(maxsize=4)`` + ``q.put()`` 阻塞。
当推理/后处理慢于采集时，队列被填满、采集线程被阻塞，整条流水线的延迟随积压
线性增长，且消费到的是"几帧之前"的历史画面——巡检场景下等于用陈旧画面做安全判定。

改造后（本模块）：

* :class:`LatestFrameSlot` —— 单槽 + 条件变量，``publish`` **永不阻塞**：
  槽内已有未消费帧则被覆盖并计入 ``dropped``，保证消费者拿到的永远是最新帧；
* :class:`BlockingFrameQueue` —— 保留改造前的有界阻塞语义，仅作对照基线，
  用于量化"阻塞式 vs 最新帧优先"的差异，避免改动后失去可比口径；
* :class:`FramePump` —— 采集线程：``source.read() → sink.publish()``，
  异常分类处理，可被 ``stop()`` 干净中断。

不变式（任一时刻成立）：``published == consumed + dropped + pending``。
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from .base import FrameSource
from .errors import DeviceUnavailableError, FrameReadError
from .models import Frame

logger = logging.getLogger(__name__)

Policy = Literal["latest", "blocking"]


# ===== 统计 =====
@dataclass
class SinkStats:
    """管道（槽/队列）侧计数。"""

    published: int = 0
    consumed: int = 0
    dropped: int = 0

    @property
    def pending(self) -> int:
        return self.published - self.consumed - self.dropped


@runtime_checkable
class FrameSink(Protocol):
    """帧管道协议：生产端 :meth:`publish`，消费端 :meth:`get`。"""

    name: str

    @property
    def is_closed(self) -> bool:
        """管道是否已关闭（关闭后 ``publish`` 返回 False、``get`` 返回 None）。"""

    def publish(self, frame: Frame) -> bool:
        """投递一帧。返回 False 表示管道已关闭、调用方应停止生产。"""

    def get(self, timeout: float | None = None) -> Frame | None:
        """取一帧；超时或已关闭返回 None。"""

    def close(self) -> None:
        """关闭管道并唤醒所有等待者。"""

    def stats(self) -> SinkStats:
        """当前计数快照。"""


# ===== 最新帧优先槽（默认策略）=====
class LatestFrameSlot:
    """单槽位、最新帧优先、丢弃陈旧帧的帧管道。

    语义：
    - ``publish`` 恒为 O(1) 且**不阻塞**；槽内已有未消费帧时，旧帧被丢弃（``dropped += 1``）；
    - ``get`` 等待"新帧到达"，取走后槽位置空；``get(timeout=0)`` 等价于非阻塞尝试；
    - ``close`` 唤醒所有等待者，已取走的帧不受影响；关闭后 ``publish`` 返回 False。

    该语义保证消费者永远处理最新画面，代价是主动丢弃来不及处理的帧——
    对巡检/告警场景，这是正确取舍：宁可少处理几帧，也不要用过期画面判定安全。
    """

    name = "latest"

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._closed = False
        self._stats = SinkStats()

    # ----- 生产 -----
    def publish(self, frame: Frame) -> bool:
        """投递一帧（非阻塞）。返回 False 表示已关闭。"""
        with self._cond:
            if self._closed:
                return False
            if self._frame is not None:
                self._stats.dropped += 1      # 覆盖即丢弃陈旧帧
            self._frame = frame
            self._stats.published += 1
            self._cond.notify()
            return True

    # ----- 消费 -----
    def get(self, timeout: float | None = None) -> Frame | None:
        """等待并取走最新帧；超时返回 None，关闭后返回 None。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                if self._frame is not None:
                    frame = self._frame
                    self._frame = None
                    self._stats.consumed += 1
                    return frame
                if self._closed:
                    return None
                if deadline is None:
                    self._cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cond.wait(remaining)

    def get_nowait(self) -> Frame | None:
        """非阻塞取帧；无帧返回 None。"""
        return self.get(timeout=0.0)

    def peek(self) -> Frame | None:
        """查看当前槽内帧但不取走；无帧返回 None。"""
        with self._cond:
            return self._frame

    # ----- 生命周期 -----
    def close(self) -> None:
        """关闭管道并唤醒等待者（幂等）。"""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def is_closed(self) -> bool:
        with self._cond:
            return self._closed

    def stats(self) -> SinkStats:
        with self._cond:
            return SinkStats(self._stats.published, self._stats.consumed, self._stats.dropped)


# ===== 阻塞队列（对照基线，保持改造前语义）=====
_SENTINEL = object()


class BlockingFrameQueue:
    """有界阻塞队列，复刻改造前 ``queue.Queue(maxsize=4)`` 的行为。

    仅作对照基线：采集线程在队列满时会被阻塞，积压会转化为端到端延迟增长。
    生产代码不应使用（除需要严格不丢帧的离线批处理场景）。
    """

    name = "blocking"

    def __init__(self, maxsize: int = 4) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = maxsize
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._closed = threading.Event()
        self._stats = SinkStats()
        self._lock = threading.Lock()

    def publish(self, frame: Frame) -> bool:
        """投递一帧；队列满时**阻塞**（这是被改造掉的旧行为）。"""
        while True:
            if self._closed.is_set():
                return False
            try:
                self._q.put(frame, timeout=0.05)
            except queue.Full:
                continue                       # 阻塞等待，直到有空位或关闭
            with self._lock:
                self._stats.published += 1
            return True

    def get(self, timeout: float | None = None) -> Frame | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                item = self._q.get(timeout=0.05)
            except queue.Empty:
                if self._closed.is_set() and self._q.empty():
                    return None
                if deadline is not None and time.monotonic() >= deadline:
                    return None
                continue
            if item is _SENTINEL:
                return None
            with self._lock:
                self._stats.consumed += 1
            return item

    def close(self) -> None:
        """置关闭标志并塞入哨兵，解除阻塞中的生产者/消费者（幂等）。"""
        if self._closed.is_set():
            return
        self._closed.set()
        for _ in range(2):                     # 两次尝试：先直接塞，队列满则腾一位再塞
            try:
                self._q.put_nowait(_SENTINEL)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    return

    @property
    def is_closed(self) -> bool:
        return self._closed.is_set()

    def stats(self) -> SinkStats:
        with self._lock:
            return SinkStats(self._stats.published, self._stats.consumed, self._stats.dropped)


def create_sink(policy: Policy = "latest", maxsize: int = 4) -> FrameSink:
    """按策略创建帧管道。

    Args:
        policy: ``latest``（默认，最新帧优先）或 ``blocking``（历史基线）。
        maxsize: 仅 ``blocking`` 生效。
    """
    if policy == "latest":
        return LatestFrameSlot()
    if policy == "blocking":
        return BlockingFrameQueue(maxsize=maxsize)
    raise ValueError(f"unknown policy: {policy!r}; expect 'latest' or 'blocking'")


# ===== 采集线程 =====
class FramePump(threading.Thread):
    """采集线程：持续 ``source.read()`` 并投递到 sink。

    Args:
        source: 已打开（或由 pump 自行打开）的采集源。
        sink: 帧管道。
        on_error: ``fail_fast`` 遇错停止；``continue`` 跳过单帧失败继续采。
        open_source: 为 True 时由 pump 负责 open/close 源。
    """

    def __init__(
        self,
        source: FrameSource,
        sink: FrameSink,
        on_error: Literal["fail_fast", "continue"] = "fail_fast",
        open_source: bool = True,
    ) -> None:
        super().__init__(name=f"pump-{source.name}", daemon=True)
        self.source = source
        self.sink = sink
        self.on_error = on_error
        self._open_source = open_source
        self._stop_event = threading.Event()
        self._finished = threading.Event()
        self.error: BaseException | None = None
        self.produced = 0
        self.failed_frames = 0
        self._lock = threading.Lock()

    # ----- 控制 -----
    def run(self) -> None:
        try:
            if self._open_source:
                self.source.open()
            self._loop()
        except DeviceUnavailableError as exc:
            self.error = exc
            logger.error("pump stopped, device unavailable: %s", exc)
        except BaseException as exc:           # noqa: BLE001 - 线程内兜底，必须记录
            self.error = exc
            logger.exception("pump crashed")
        finally:
            if self._open_source:
                try:
                    self.source.close()
                except Exception:              # noqa: BLE001 - 关闭阶段异常不覆盖原始错误
                    logger.exception("pump: source close failed")
            self._finished.set()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self.source.read()
            except FrameReadError as exc:
                with self._lock:
                    self.failed_frames += 1
                logger.warning("read frame failed: %s", exc)
                if self.on_error == "fail_fast":
                    self.error = exc
                    return
                continue
            if frame is None:                  # 源耗尽（视频读完 / 有限合成源）
                logger.info("source exhausted: %s", self.source.name)
                return
            if not self.sink.publish(frame):
                return                         # 管道已关闭
            with self._lock:
                self.produced += 1

    def stop(self, timeout: float = 2.0) -> bool:
        """请求停止并等待线程退出。返回是否已停止。"""
        self._stop_event.set()
        self.join(timeout)
        return not self.is_alive()

    @property
    def finished(self) -> bool:
        return self._finished.is_set() or not self.is_alive()

    def wait(self, timeout: float | None = None) -> bool:
        """等待采集自然结束（源耗尽或出错）。"""
        return self._finished.wait(timeout)
