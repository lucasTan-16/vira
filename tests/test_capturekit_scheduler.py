"""帧调度测试：最新帧优先 / 丢弃陈旧帧 / 非阻塞 / 停机语义。

覆盖 D1 的改造目标：
1. ``publish`` 恒不阻塞——慢消费者不会把采集线程拖住；
2. 槽内未消费的旧帧被覆盖并计入 ``dropped``，消费者只拿最新帧；
3. 计数不变式 ``published == consumed + dropped + pending`` 恒成立；
4. ``close`` 能唤醒所有等待者，不产生永久阻塞的线程。
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from capturekit import (
    BlockingFrameQueue,
    Frame,
    FramePump,
    LatestFrameSlot,
    SinkStats,
    SyntheticSource,
    create_sink,
)


def make_frame(seq: int) -> Frame:
    return Frame(
        seq=seq,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        ts_capture=time.perf_counter(),
        source="test",
    )


# ===== 最新帧优先槽 =====
def test_publish_is_non_blocking_under_no_consumer():
    """无消费者时连续投递 1000 帧，publish 总耗时必须是毫秒级（不能阻塞）。"""
    slot = LatestFrameSlot()
    started = time.perf_counter()
    for i in range(1000):
        assert slot.publish(make_frame(i)) is True
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5, f"publish blocked: {elapsed:.3f}s for 1000 frames"
    assert slot.stats().published == 1000


def test_stale_frames_are_dropped_and_latest_wins():
    slot = LatestFrameSlot()
    for i in range(10):
        slot.publish(make_frame(i))

    stats = slot.stats()
    assert stats.published == 10
    assert stats.dropped == 9                  # 9 帧未被消费即被覆盖
    assert stats.pending == 1

    frame = slot.get(timeout=0.1)
    assert frame.seq == 9                      # 拿到的是最新帧，而不是最早帧
    assert slot.stats().consumed == 1
    assert slot.stats().pending == 0


def test_get_with_timeout_returns_none():
    slot = LatestFrameSlot()
    started = time.perf_counter()
    assert slot.get(timeout=0.05) is None
    elapsed = time.perf_counter() - started
    assert 0.04 <= elapsed < 0.5                # 到点返回，不永久阻塞


def test_get_nowait_and_peek():
    slot = LatestFrameSlot()
    assert slot.get_nowait() is None
    slot.publish(make_frame(1))
    assert slot.peek().seq == 1                 # peek 不取走
    assert slot.get_nowait().seq == 1
    assert slot.get_nowait() is None


def test_consumer_blocked_until_frame_arrives():
    slot = LatestFrameSlot()
    received: list[int] = []

    def consume():
        frame = slot.get(timeout=2.0)
        received.append(frame.seq if frame else -1)

    worker = threading.Thread(target=consume)
    worker.start()
    time.sleep(0.05)
    slot.publish(make_frame(42))
    worker.join(2.0)
    assert received == [42]


def test_close_wakes_blocked_consumer():
    slot = LatestFrameSlot()
    result: list[object] = []

    def consume():
        result.append(slot.get(timeout=5.0))

    worker = threading.Thread(target=consume)
    worker.start()
    time.sleep(0.05)
    slot.close()
    worker.join(1.0)
    assert not worker.is_alive(), "close() failed to wake blocked consumer"
    assert result == [None]


def test_publish_after_close_returns_false():
    slot = LatestFrameSlot()
    slot.close()
    assert slot.publish(make_frame(1)) is False
    assert slot.is_closed is True


def test_closed_slot_still_drains_pending_frame():
    slot = LatestFrameSlot()
    slot.publish(make_frame(7))
    slot.close()
    assert slot.get(timeout=0.1).seq == 7       # 关闭不丢最后一帧
    assert slot.get(timeout=0.05) is None


def test_counter_invariant_under_concurrency():
    """多生产者 + 单消费者并发下，计数不变式必须始终成立。"""
    slot = LatestFrameSlot()
    stop = threading.Event()
    errors: list[str] = []

    def producer(offset: int):
        i = 0
        while not stop.is_set():
            slot.publish(make_frame(offset + i))
            i += 1

    def consumer():
        while not stop.is_set():
            frame = slot.get(timeout=0.01)
            if frame is not None:
                time.sleep(0.001)

    def checker():
        while not stop.is_set():
            s = slot.stats()
            if s.published != s.consumed + s.dropped + s.pending:
                errors.append(f"broken invariant: {s}")
            time.sleep(0.005)

    threads = [
        threading.Thread(target=producer, args=(0,)),
        threading.Thread(target=producer, args=(100000,)),
        threading.Thread(target=consumer),
        threading.Thread(target=checker),
    ]
    for t in threads:
        t.start()
    time.sleep(0.5)
    stop.set()
    for t in threads:
        t.join(2.0)

    s = slot.stats()
    assert not errors, errors[:3]
    assert s.published == s.consumed + s.dropped + s.pending
    assert s.dropped > 0, "slow consumer should cause drops under latest policy"


# ===== 阻塞队列（对照基线）=====
def test_blocking_queue_preserves_old_semantics():
    """基线行为：不丢帧、按序消费——但队列满时生产者会阻塞。"""
    sink = BlockingFrameQueue(maxsize=2)
    sink.publish(make_frame(0))
    sink.publish(make_frame(1))
    assert sink.stats().published == 2

    blocked = threading.Event()

    def producer():
        sink.publish(make_frame(2))            # 队列已满，此处阻塞
        blocked.set()

    worker = threading.Thread(target=producer, daemon=True)
    worker.start()
    assert not blocked.wait(0.1), "baseline queue should block when full"

    assert sink.get(timeout=0.2).seq == 0      # 消费腾位后生产者才恢复
    assert blocked.wait(1.0)
    assert [sink.get(timeout=0.2).seq, sink.get(timeout=0.2).seq] == [1, 2]


def test_blocking_queue_close_unblocks_producer():
    sink = BlockingFrameQueue(maxsize=1)
    sink.publish(make_frame(0))

    done = threading.Event()

    def producer():
        sink.publish(make_frame(1))
        done.set()

    threading.Thread(target=producer, daemon=True).start()
    time.sleep(0.1)
    sink.close()
    assert done.wait(1.0), "close() must unblock a producer stuck on a full queue"


def test_create_sink_dispatch():
    assert isinstance(create_sink("latest"), LatestFrameSlot)
    assert isinstance(create_sink("blocking", maxsize=4), BlockingFrameQueue)
    with pytest.raises(ValueError):
        create_sink("nonexistent")            # type: ignore[arg-type]


def test_stats_pending_property():
    stats = SinkStats(published=10, consumed=3, dropped=6)
    assert stats.pending == 1


# ===== 采集线程 =====
def test_pump_publishes_all_frames_of_finite_source():
    source = SyntheticSource(width=32, height=32, fps=0, total=20)
    slot = LatestFrameSlot()
    pump = FramePump(source, slot)
    pump.start()
    assert pump.wait(5.0), "pump did not finish"
    pump.stop()

    assert pump.produced == 20
    assert slot.stats().published == 20
    assert source.stats.frames_read == 20
    assert pump.error is None


def test_pump_stop_unblocks_promptly():
    source = SyntheticSource(width=32, height=32, fps=200)
    slot = LatestFrameSlot()
    pump = FramePump(source, slot)
    pump.start()
    time.sleep(0.1)
    assert pump.stop(timeout=2.0) is True, "pump failed to stop"
    assert slot.stats().published > 0


def test_pump_surfaces_device_error():
    from capturekit import CameraConfig, CameraSource

    pump = FramePump(CameraSource(CameraConfig(index=99)), LatestFrameSlot())
    pump.start()
    pump.wait(3.0)
    assert isinstance(pump.error, Exception)
    assert pump.produced == 0


def test_pump_continue_mode_survives_read_errors():
    """on_error=continue：单帧失败不终止采集（对应偶发 USB 抖动）。"""
    source = SyntheticSource(width=16, height=16, fps=0, total=6, fail_every=3)
    source.max_read_failures = 10
    slot = LatestFrameSlot()
    pump = FramePump(source, slot, on_error="continue")
    pump.start()
    pump.wait(5.0)
    pump.stop()

    assert pump.failed_frames >= 1
    assert slot.stats().published >= 3
