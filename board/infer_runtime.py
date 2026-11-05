"""板端推理运行时：采集 → 推理(NPU) → 后处理(CPU) 的工程化流水线。

与改造前 ``camera_infer.py`` / ``pipeline_infer.py`` 的差别：

1. **采集由 capturekit 抽象层负责**——不再各脚本自己 ``cv2.VideoCapture``、
   各写一份 letterbox；
2. **帧在管道中"最新优先、可丢弃"**——慢消费不会反压采集（原队列满即阻塞）；
3. **停顿与统计可观测**——统一输出产帧/推理/后处理/丢弃/端到端延迟，
   板端回放录像与开发机合成源走同一套代码；
4. **可测试**——``PipelineRunner`` 只依赖 ``detector.detect(frame)`` 协议，
   没有 NPU 也能用假检测器验证并发与调度行为。

用法（见 board/camera_infer.py、board/pipeline_infer.py 两个 CLI）::

    runner = PipelineRunner(source, VisionRuntime(model_path))
    report = runner.run(max_frames=300)
    print(report.format_text())
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from capturekit import (  # noqa: E402 - 需要先修好 sys.path
    CaptureReport,
    Frame,
    FramePump,
    FrameSink,
    FrameSource,
    LatencyTracker,
    Policy,
    create_sink,
)

logger = logging.getLogger(__name__)

PostprocessFn = Callable[[Any], Any]


class Detector(Protocol):
    """推理后端协议：输入一帧，返回原始输出（交给后处理）。"""

    def detect(self, frame: Frame) -> Any:
        ...


# ===== NPU 推理实现 =====
class VisionRuntime:
    """RKNN 推理封装（CORE_0 单核）。

    延迟导入 ``rknnlite``：开发机 / CI 无 NPU 环境也能 import 本项目并跑非 NPU 测试。
    """

    def __init__(
        self,
        model_path: str,
        core_mask: int | None = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
    ) -> None:
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self._rk = None
        self._core_mask = core_mask
        self._postprocess = None

    def load(self) -> "VisionRuntime":
        """加载并初始化 NPU 运行时。"""
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"rknn model not found: {self.model_path}")
        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:  # pragma: no cover - 板端才有
            raise RuntimeError(
                "rknnlite not available; run this on the board (RK3588) or pass a "
                "custom detector to PipelineRunner"
            ) from exc

        from postprocess import decode_three_head, nms

        self._postprocess = (decode_three_head, nms)
        rk = RKNNLite()
        if rk.load_rknn(self.model_path) != 0:
            raise RuntimeError(f"load_rknn failed: {self.model_path}")
        rk.init_runtime(
            core_mask=self._core_mask if self._core_mask is not None else RKNNLite.NPU_CORE_0
        )
        self._rk = rk
        logger.info("rknn runtime ready: %s", os.path.basename(self.model_path))
        return self

    def detect(self, frame: Frame) -> list:
        """NPU 推理 + decode + NMS，返回检测框列表。"""
        if self._rk is None:
            raise RuntimeError("VisionRuntime not loaded; call load() first")
        decode_three_head, nms = self._postprocess
        out = self._rk.inference(inputs=[frame.image[None, ...]], data_format="nhwc")
        return nms(decode_three_head(out), conf_thres=self.conf_thres, iou_thres=self.iou_thres)

    def warmup(self, frames: list[Frame], rounds: int = 5) -> None:
        """预热：让 NPU 与 CPU 后处理都进入稳态，避免首帧抖动计入统计。"""
        if not frames:
            return
        for _ in range(rounds):
            for frame in frames:
                self.detect(frame)

    def release(self) -> None:
        if self._rk is not None:
            self._rk.release()
            self._rk = None


# ===== 流水线报告 =====
@dataclass(frozen=True)
class PipelineReport:
    """一次连续推理会话的统计。"""

    policy: str
    source: str
    wall_s: float
    frames_captured: int
    frames_inferred: int
    frames_postprocessed: int
    dropped_capture: int
    dropped_infer: int
    read_errors: int
    latency: dict[str, float] = field(default_factory=dict)

    @property
    def capture_fps(self) -> float:
        return self.frames_captured / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def infer_fps(self) -> float:
        return self.frames_inferred / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def post_fps(self) -> float:
        """稳态吞吐：后处理线程的完成速率（= 流水线吞吐上限）。"""
        return self.frames_postprocessed / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def dropped_total(self) -> int:
        return self.dropped_capture + self.dropped_infer

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "source": self.source,
            "wall_s": round(self.wall_s, 3),
            "frames_captured": self.frames_captured,
            "frames_inferred": self.frames_inferred,
            "frames_postprocessed": self.frames_postprocessed,
            "dropped_capture": self.dropped_capture,
            "dropped_infer": self.dropped_infer,
            "dropped_total": self.dropped_total,
            "read_errors": self.read_errors,
            "capture_fps": round(self.capture_fps, 2),
            "infer_fps": round(self.infer_fps, 2),
            "post_fps": round(self.post_fps, 2),
            "latency": self.latency,
        }

    def as_capture_report(self) -> CaptureReport:
        """转换为采集层报告，便于与 ``capturekit.diagnostics`` 输出对齐。"""
        return CaptureReport(
            source=self.source,
            policy=self.policy,
            window_s=self.wall_s,
            frames_produced=self.frames_captured,
            frames_published=self.frames_inferred + self.dropped_capture,
            frames_consumed=self.frames_inferred,
            frames_dropped=self.dropped_capture,
            read_errors=self.read_errors,
            latency=self.latency,
        )

    def format_text(self) -> str:
        lines = [
            f"采集源      : {self.source}",
            f"调度策略    : {self.policy}",
            f"墙钟        : {self.wall_s:.3f} s",
            f"采集        : {self.frames_captured} 帧（{self.capture_fps:.1f} FPS）",
            f"推理        : {self.frames_inferred} 帧（{self.infer_fps:.1f} FPS）",
            f"后处理完成  : {self.frames_postprocessed} 帧（吞吐 {self.post_fps:.1f} FPS）",
            f"丢弃帧      : 采集段 {self.dropped_capture} + 推理段 {self.dropped_infer}"
            f" = {self.dropped_total}",
            f"读失败      : {self.read_errors}",
        ]
        if self.latency:
            lines.append(
                "端到端延迟  : avg {avg_ms:.2f} ms / p95 {p95_ms:.2f} ms / max {max_ms:.2f} ms".format(
                    **self.latency
                )
            )
        return "\n".join(lines)


# ===== 流水线 =====
class PipelineRunner:
    """三线程流水线：采集(pump) → 推理(NPU) → 后处理(CPU)。

    Args:
        source: 采集源（capturekit FrameSource）。
        detector: 推理后端，需实现 ``detect(frame) -> Any``。
        postprocess: 后处理函数，默认丢弃检测结果（纯吞吐用途）。
        policy: 帧调度策略，默认 ``latest``（最新帧优先）。
        maxsize: ``policy="blocking"`` 时的队列长度（对照基线用）。
    """

    def __init__(
        self,
        source: FrameSource,
        detector: Detector,
        postprocess: PostprocessFn | None = None,
        policy: Policy = "latest",
        maxsize: int = 4,
    ) -> None:
        self.source = source
        self.detector = detector
        self.policy = policy
        self.postprocess = postprocess or (lambda result: result)
        self.in_sink: FrameSink = create_sink(policy, maxsize=maxsize)
        self.out_sink: FrameSink = create_sink(policy, maxsize=maxsize)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._inferred = 0
        self._posted = 0
        self._infer_error: BaseException | None = None
        self._post_error: BaseException | None = None
        self._latency = LatencyTracker()
        self._pump = FramePump(source, self.in_sink)

    # ----- worker -----
    def _infer_worker(self) -> None:
        while not self._stop.is_set():
            frame = self.in_sink.get(timeout=0.1)
            if frame is None:
                if self._stop.is_set() or self.in_sink.is_closed or self._pump.finished:
                    break
                continue
            try:
                result = self.detector.detect(frame)
            except BaseException as exc:  # noqa: BLE001 - 线程内必须兜底并上报
                self._infer_error = exc
                logger.exception("infer worker failed")
                break
            self.out_sink.publish((frame, result))
            with self._lock:
                self._inferred += 1

    def _post_worker(self, max_frames: int | None) -> None:
        while not self._stop.is_set():
            item = self.out_sink.get(timeout=0.1)
            if item is None:
                if self._stop.is_set() or self.out_sink.is_closed or self._pump.finished:
                    break
                continue
            frame, result = item
            try:
                self.postprocess(result)
            except BaseException as exc:  # noqa: BLE001
                self._post_error = exc
                logger.exception("post worker failed")
                break
            with self._lock:
                self._posted += 1
                posted = self._posted
            self._latency.record(frame.age_ms())
            if max_frames is not None and posted >= max_frames:
                self._stop.set()

    # ----- 主流程 -----
    def run(self, max_frames: int | None = 300, duration: float | None = None) -> PipelineReport:
        """运行流水线直到处理满 ``max_frames`` 帧或达到 ``duration`` 秒。

        Args:
            max_frames: 后处理完成的帧数上限；``None`` 表示不限。
            duration: 最长运行秒数；``None`` 表示不限（须给 ``max_frames``）。
        """
        if max_frames is None and duration is None:
            raise ValueError("run() needs max_frames or duration to terminate")

        started = time.perf_counter()
        self._pump.start()
        threads = [
            threading.Thread(target=self._infer_worker, name="infer", daemon=True),
            threading.Thread(target=self._post_worker, args=(max_frames,), name="post", daemon=True),
        ]
        for t in threads:
            t.start()

        try:
            while not self._stop.is_set():
                if duration is not None and time.perf_counter() - started >= duration:
                    break
                if self._infer_error or self._post_error or self._pump.error:
                    break
                time.sleep(0.02)
        finally:
            self._stop.set()
            # 先关管道解除可能的阻塞投递，再停采集线程，避免停机挂死
            self.in_sink.close()
            self.out_sink.close()
            self._pump.stop(timeout=3.0)
            for t in threads:
                t.join(2.0)
            self.source.close()

        wall = time.perf_counter() - started
        cap_stats = self.in_sink.stats()
        out_stats = self.out_sink.stats()
        error = self._infer_error or self._post_error or self._pump.error
        if error is not None:
            logger.error("pipeline stopped by error: %r", error)

        return PipelineReport(
            policy=self.policy,
            source=self.source.name,
            wall_s=wall,
            frames_captured=self.source.stats.frames_read,
            frames_inferred=self._inferred,
            frames_postprocessed=self._posted,
            dropped_capture=cap_stats.dropped,
            dropped_infer=out_stats.dropped,
            read_errors=self.source.stats.read_errors,
            latency=self._latency.snapshot(),
        )

    @property
    def error(self) -> BaseException | None:
        """运行中出现的首个致命错误（None 表示正常结束）。"""
        return self._infer_error or self._post_error or self._pump.error
