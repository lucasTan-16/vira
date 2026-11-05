"""采集层自检 CLI：板端/开发机一行命令判断"能不能采、采多快、丢不丢帧"。

用法::

    python3 -m capturekit.diagnostics                    # 探测默认摄像头
    python3 -m capturekit.diagnostics --index 1 -n 30    # 指定设备号，采 30 帧测速
    python3 -m capturekit.diagnostics --config cfg.json  # 按配置文件探测
    python3 -m capturekit.diagnostics --selftest         # 跳过硬件，仅跑调度自检

退出码：0 硬件可用 / 1 硬件不可用（自检结论正常时） / 2 自检失败。
摄像头损坏场景（C8）下本工具会明确给出 *DeviceUnavailableError* 原因，
而不是让上层脚本在 ``cv2.VideoCapture`` 里静默拿到空帧。
"""
from __future__ import annotations

import argparse
import glob
import logging
import platform
import sys
import time

from .config import CameraConfig, CaptureConfig
from .errors import CaptureError, DeviceUnavailableError
from .metrics import LatencyTracker
from .models import Frame  # noqa: F401 - 类型标注用
from .registry import create_source
from .scheduler import FramePump, LatestFrameSlot

logger = logging.getLogger("capturekit.diagnostics")


def _print_env() -> None:
    print("== 环境 ==")
    print(f"platform : {platform.platform()}")
    print(f"python   : {platform.python_version()}")
    try:
        import cv2

        print(f"opencv   : {cv2.__version__}")
    except Exception as exc:  # noqa: BLE001 - 诊断场景要求尽力输出
        print(f"opencv   : 不可用 ({exc})")
    devices = sorted(glob.glob("/dev/video*"))
    print(f"v4l2设备 : {devices if devices else '无 (/dev/video* 不存在)'}")


def _probe_camera(config: CaptureConfig, frames: int) -> int:
    """打开摄像头采 ``frames`` 帧并输出实测帧率。返回退出码。"""
    source = create_source(config)
    print("\n== 摄像头探测 ==")
    print(f"配置     : {config.summary()}")
    started = time.perf_counter()
    try:
        source.open()
    except DeviceUnavailableError as exc:
        print(f"结论     : 设备不可用 —— {exc}")
        print("建议     : 检查 USB 连接 / dmesg | grep -i uvc / ls -l /dev/video*")
        return 1
    except CaptureError as exc:
        print(f"结论     : 打开失败 —— {exc}")
        return 1

    try:
        last_shape = None
        for _ in range(frames):
            frame = source.read()
            if frame is None:
                break
            last_shape = frame.meta.get("raw_shape")
        elapsed = time.perf_counter() - started
        stats = source.stats.as_dict()
        fps = stats["frames_read"] / elapsed if elapsed > 0 else 0.0
        print(f"读帧     : {stats['frames_read']} 帧 / {elapsed:.3f} s → {fps:.1f} FPS")
        print(f"读失败   : {stats['read_errors']}")
        if last_shape:
            print(f"原始尺寸 : {last_shape[1]}x{last_shape[0]}（HWC {tuple(last_shape)}）")
        ok = stats["frames_read"] > 0
        print(f"结论     : {'可用' if ok else '打开成功但取不到帧'}")
        return 0 if ok else 1
    finally:
        source.close()


def _selftest() -> int:
    """无硬件自检：合成源 + 慢消费，验证"丢弃陈旧帧"确实生效。"""
    print("\n== 调度自检（合成源，不需要摄像头）==")
    cfg = CaptureConfig.from_dict(
        {"kind": "synthetic", "synthetic": {"width": 320, "height": 240, "fps": 120.0}}
    )
    sink = LatestFrameSlot()
    pump = FramePump(create_source(cfg), sink)
    pump.start()
    latency = LatencyTracker()
    consumed = 0
    deadline = time.perf_counter() + 0.6
    try:
        while time.perf_counter() < deadline:
            frame = sink.get(timeout=0.2)
            if frame is None:
                break
            consumed += 1
            latency.record(frame.age_ms())
            time.sleep(0.02)                    # 模拟每帧 20ms 的推理耗时
    finally:
        pump.stop()
        sink.close()

    stats = sink.stats()
    pump_stats = pump.source.stats.as_dict()
    print(f"产帧     : {pump_stats['frames_read']}")
    print(f"发布/消费: {stats.published} / {consumed}")
    print(f"丢弃     : {stats.dropped}（丢弃率 {stats.dropped / stats.published * 100:.1f}%"
          if stats.published else "丢弃     : 0")
    print(f"消费延迟 : {latency.snapshot()}")

    ok = stats.published > 0 and stats.dropped > 0 and consumed > 0
    print(f"结论     : {'通过（已丢弃陈旧帧，消费者拿到最新帧）' if ok else '未通过'}")
    if not ok:
        print("可能原因 : 机器过慢导致产帧不足，或调度实现有误")
    return 0 if ok else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="capturekit 采集层自检")
    parser.add_argument("--index", type=int, default=None, help="摄像头设备号（默认 0）")
    parser.add_argument("-n", "--frames", type=int, default=30, help="采样帧数，默认 30")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（JSON/YAML）")
    parser.add_argument("--selftest", action="store_true", help="跳过硬件，仅跑调度自检")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    _print_env()

    if args.selftest:
        return _selftest()

    if args.config:
        config = CaptureConfig.from_file(args.config)
    else:
        index = 0 if args.index is None else args.index
        config = CaptureConfig(kind="camera", camera=CameraConfig(index=index))

    if config.kind != "camera":
        print(f"\n非摄像头配置（kind={config.kind}），改为调度自检")
        return _selftest()

    code = _probe_camera(config, max(1, args.frames))
    print("\n== 调度自检（合成源）==")
    _selftest()
    return code


if __name__ == "__main__":
    sys.exit(main())
