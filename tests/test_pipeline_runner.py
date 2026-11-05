"""板端流水线（infer_runtime）测试：无需 NPU 与摄像头。

用假检测器 + 合成源验证：
- ``latest`` 策略下慢检测不会反压采集（产帧远多于推理）；
- ``blocking`` 策略复现改造前的阻塞行为（不丢帧，但采集被消费速率绑死）；
- 推理异常会被上报而不是静默吞掉；
- 采集层抽象化后，Agent 工具不再直接依赖 cv2 摄像头调用。
"""
from __future__ import annotations

import os
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOARD = os.path.join(ROOT, "board")
for _p in (ROOT, BOARD):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from capturekit import Frame, SyntheticSource          # noqa: E402
from infer_runtime import PipelineReport, PipelineRunner, VisionRuntime  # noqa: E402


class FakeDetector:
    """模拟 NPU 推理耗时；可选在第 N 次调用抛错。"""

    def __init__(self, delay: float = 0.02, fail_at: int | None = None) -> None:
        self.delay = delay
        self.fail_at = fail_at
        self.calls = 0

    def detect(self, frame: Frame) -> list:
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise RuntimeError("npu boom")
        time.sleep(self.delay)
        return []


def make_runner(policy: str, detector: FakeDetector, maxsize: int = 4, total: int = 0) -> PipelineRunner:
    source = SyntheticSource(width=64, height=48, fps=0, total=total)
    return PipelineRunner(source, detector, policy=policy, maxsize=maxsize)


def test_latest_policy_decouples_capture_from_slow_detector():
    runner = make_runner("latest", FakeDetector(delay=0.02))
    report = runner.run(max_frames=15)

    assert report.frames_postprocessed == 15
    assert report.frames_inferred >= 15
    assert report.frames_captured > report.frames_inferred * 2, report.to_dict()
    assert report.dropped_capture > 0
    assert report.infer_fps > 0 and report.post_fps > 0
    assert report.latency["samples"] == 15


def test_blocking_policy_reproduces_backpressure():
    runner = make_runner("blocking", FakeDetector(delay=0.02), maxsize=4)
    report = runner.run(max_frames=15)

    assert report.frames_postprocessed == 15
    assert report.dropped_capture == 0
    # 队列长度 4 → 产帧最多领先 4 帧（+1 在途），采集被消费速率绑死
    assert report.frames_captured <= report.frames_inferred + 6, report.to_dict()


def test_capture_throughput_gap_between_policies():
    fast_latest = make_runner("latest", FakeDetector(delay=0.01)).run(max_frames=10)
    baseline = make_runner("blocking", FakeDetector(delay=0.01), maxsize=4).run(max_frames=10)
    assert fast_latest.frames_captured > baseline.frames_captured * 2


def test_detector_failure_is_reported_not_swallowed():
    runner = make_runner("latest", FakeDetector(delay=0.001, fail_at=3))
    report = runner.run(max_frames=50)

    assert isinstance(runner.error, RuntimeError)
    assert report.frames_postprocessed < 50
    assert report.wall_s > 0
    assert report.frames_postprocessed == report.frames_inferred  # 出错后不再有新推理


def test_duration_bound_run_terminates():
    runner = make_runner("latest", FakeDetector(delay=0.005))
    started = time.perf_counter()
    report = runner.run(max_frames=None, duration=0.3)
    elapsed = time.perf_counter() - started

    assert 0.25 <= elapsed < 3.0
    assert report.frames_postprocessed > 0
    assert runner.error is None


def test_run_requires_termination_condition():
    runner = make_runner("latest", FakeDetector())
    with pytest.raises(ValueError):
        runner.run(max_frames=None, duration=None)


def test_report_serialisation():
    report = PipelineReport(
        policy="latest",
        source="synthetic",
        wall_s=2.0,
        frames_captured=400,
        frames_inferred=80,
        frames_postprocessed=80,
        dropped_capture=320,
        dropped_infer=0,
        read_errors=0,
        latency={"avg_ms": 5.0, "p95_ms": 9.0, "max_ms": 12.0, "samples": 80},
    )
    payload = report.to_dict()
    assert payload["capture_fps"] == 200.0
    assert payload["post_fps"] == 40.0
    assert payload["dropped_total"] == 320
    text = report.format_text()
    assert "丢弃帧" in text and "端到端延迟" in text
    cap = report.as_capture_report()
    assert cap.policy == "latest" and cap.frames_dropped == 320


def test_vision_runtime_reports_missing_model_clearly():
    with pytest.raises(RuntimeError) as err:
        VisionRuntime("/nonexistent/best.rknn").load()
    assert "not found" in str(err.value)


@pytest.mark.skipif(os.path.exists("/dev/video0"), reason="本机有摄像头，不适用无设备路径")
def test_agent_tool_camera_path_uses_capture_abstraction():
    """工具层采集应走 capturekit：无设备时抛出可读 RuntimeError 而非 cv2 静默失败。"""
    from agent.tools import _capture_frame

    with pytest.raises(RuntimeError) as err:
        _capture_frame()
    assert "camera" in str(err.value).lower()
