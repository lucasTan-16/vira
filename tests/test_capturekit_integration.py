"""端到端对照：阻塞式（改造前）vs 最新帧优先（改造后）。

用合成源 + 人为放慢的消费者，量化 D1 问题与改造收益，不依赖摄像头/模型/NPU：

指标含义
- ``produced``   采集线程实际读到的帧数 → 反映"采集是否被推理拖住"
- ``consumed``   下游处理帧数
- ``dropped``    被丢弃的陈旧帧数（仅最新帧策略会有）
- ``avg/p95 age``消费时帧的年龄 = 端到端"画面新鲜度"延迟
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from capturekit import (
    CaptureReport,
    FramePump,
    LatencyTracker,
    SyntheticSource,
    create_sink,
)

PRODUCE_FPS = 200.0      # 采集能力：5 ms/帧
CONSUMER_MS = 50.0       # 下游处理：50 ms/帧（模拟 NPU 推理 + 后处理）
DURATION_S = 0.8


@dataclass
class SessionResult:
    policy: str
    produced: int
    consumed: int
    dropped: int
    avg_ms: float
    p95_ms: float
    max_ms: float
    window_s: float

    def as_report(self) -> CaptureReport:
        return CaptureReport(
            source="synthetic",
            policy=self.policy,
            window_s=self.window_s,
            frames_produced=self.produced,
            frames_published=self.produced,
            frames_consumed=self.consumed,
            frames_dropped=self.dropped,
            read_errors=0,
            latency={"avg_ms": round(self.avg_ms, 3), "p95_ms": round(self.p95_ms, 3),
                     "max_ms": round(self.max_ms, 3), "samples": self.consumed},
        )


def run_session(policy: str, maxsize: int = 4) -> SessionResult:
    """跑一次采集会话：独立线程产帧，主线程按固定耗时消费。"""
    source = SyntheticSource(width=64, height=48, fps=PRODUCE_FPS)
    sink = create_sink(policy, maxsize=maxsize)      # type: ignore[arg-type]
    pump = FramePump(source, sink)
    latency = LatencyTracker()
    consumed = 0

    pump.start()
    started = time.perf_counter()
    try:
        while True:
            if time.perf_counter() - started >= DURATION_S:
                break
            frame = sink.get(timeout=0.2)
            if frame is None:
                break
            latency.record(frame.age_ms())
            consumed += 1
            time.sleep(CONSUMER_MS / 1000.0)
    finally:
        sink.close()                                 # 先解除可能的阻塞投递
        pump.stop(timeout=3.0)

    window = time.perf_counter() - started
    stats = sink.stats()
    snap = latency.snapshot()
    return SessionResult(
        policy=policy,
        produced=source.stats.frames_read,
        consumed=consumed,
        dropped=stats.dropped,
        avg_ms=snap["avg_ms"],
        p95_ms=snap["p95_ms"],
        max_ms=snap["max_ms"],
        window_s=window,
    )


@pytest.fixture(scope="module")
def sessions():
    return {
        "latest": run_session("latest"),
        "blocking": run_session("blocking", maxsize=4),
    }


def test_latest_policy_keeps_capture_rate_independent_of_consumer(sessions):
    """核心结论：慢消费者不应拖慢采集。

    阻塞策略下队列填满后采集线程被 ``put`` 卡住，产帧率被压到消费速率附近；
    最新帧策略下采集线程全速产帧，多余帧直接丢弃。
    """
    latest, blocking = sessions["latest"], sessions["blocking"]
    expected = DURATION_S * PRODUCE_FPS * 0.5      # 留 50% 余量吸收调度抖动
    assert latest.produced > expected, f"latest produced too few frames: {latest}"
    assert latest.produced > blocking.produced * 1.5, (
        f"capture was throttled by consumer: latest={latest.produced}, "
        f"blocking={blocking.produced}"
    )


def test_latest_policy_drops_stale_frames(sessions):
    """慢消费者场景下必须发生丢弃——这正是"不积压"的代价与证据。"""
    latest, blocking = sessions["latest"], sessions["blocking"]
    assert latest.dropped > 0
    assert blocking.dropped == 0                   # 旧行为：不丢帧，改为积压


def test_latest_policy_keeps_frames_fresh(sessions):
    """消费者拿到的帧更"新"：平均年龄更低。"""
    latest, blocking = sessions["latest"], sessions["blocking"]
    assert latest.consumed > 0 and blocking.consumed > 0
    assert latest.avg_ms < blocking.avg_ms, (
        f"latest not fresher: latest={latest.avg_ms:.2f}ms blocking={blocking.avg_ms:.2f}ms"
    )
    assert latest.max_ms < blocking.max_ms * 2.0


def test_session_report_is_serialisable(sessions):
    report = sessions["latest"].as_report()
    payload = report.to_dict()
    assert payload["drop_ratio"] > 0
    assert payload["producer_fps"] > payload["consumer_fps"]
    text = report.format_text()
    assert "丢弃陈旧帧" in text and "消费延迟" in text


def test_report_invariant():
    report = CaptureReport(
        source="synthetic",
        policy="latest",
        window_s=1.0,
        frames_produced=100,
        frames_published=100,
        frames_consumed=20,
        frames_dropped=79,
        read_errors=0,
    )
    assert report.pending == 1
    assert report.drop_ratio == pytest.approx(0.79)
    assert report.consumer_fps == pytest.approx(20.0)
