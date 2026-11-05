#!/usr/bin/env python3
"""板端摄像头冒烟测试：经 capturekit 抽象层打开设备并读若干帧。

改造后不再直接用 ``cv2.VideoCapture``：设备打开、MJPG/分辨率下发、驱动缓冲深度、
读帧失败重试与异常语义全部由采集抽象层统一处理，与 ``camera_infer.py`` 同一条链路，
避免"冒烟脚本能跑、主程序不能跑"的两套采集逻辑。

用法::

    python3 board/test_camera.py                 # /dev/video0, 640x480 MJPG
    python3 board/test_camera.py --index 1 --frames 10
    python3 board/test_camera.py --list          # 只打印设备探测结果
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from capturekit import (  # noqa: E402
    CameraConfig,
    CameraSource,
    CaptureError,
    FrameReadError,
    absence_hint,
    has_camera,
    probe_camera_devices,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="板端摄像头冒烟测试（capturekit 链路）")
    parser.add_argument("--index", type=int, default=0, help="设备编号，0 = /dev/video0")
    parser.add_argument("--frames", type=int, default=5, help="读取帧数")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fourcc", default="MJPG", help="像素格式；YUYV 高分辨率帧率低")
    parser.add_argument("--fps", type=int, default=30, help="目标采集帧率")
    parser.add_argument("--buffer-size", type=int, default=1, help="驱动缓冲帧数，1=只保留最新")
    parser.add_argument("--max-read-failures", type=int, default=5,
                        help="连续读失败上限，超过判定设备故障")
    parser.add_argument("--list", action="store_true", help="仅打印设备探测结果")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    devices = probe_camera_devices()
    print("探测到的视频设备: " + (", ".join(devices) if devices else "无"))
    if args.list:
        if not devices:
            print(absence_hint(), file=sys.stderr)
        return 0 if has_camera() else 2

    config = CameraConfig(
        index=args.index,
        width=args.width,
        height=args.height,
        fourcc=args.fourcc,
        fps=args.fps,
        buffer_size=args.buffer_size,
        max_read_failures=args.max_read_failures,
    )
    source = CameraSource(config)

    stamps: list[float] = []
    try:
        source.open()
        print(f"已打开 {source.name}")
        for seq in range(args.frames):
            frame = source.read()
            if frame is None:
                print("read() 返回 None（源已关闭或耗尽）")
                break
            if seq == 0:
                height, width = frame.image.shape[:2]
                print(f"首帧: {width}x{height} seq={frame.seq} 预处理后张量形状={frame.image.shape}")
            stamps.append(frame.captured_at)
            print(f"  帧 {seq}: 采集时刻距现在 {frame.age_ms():.1f} ms")
    except FrameReadError as exc:
        print(f"读帧失败: {exc}", file=sys.stderr)
        print("排查: 设备是否被占用 / 降分辨率或改 YUYV / 换 USB 口供电", file=sys.stderr)
        return 3
    except CaptureError as exc:
        print(f"设备不可用: {exc}", file=sys.stderr)
        print(absence_hint(), file=sys.stderr)
        return 2
    finally:
        source.close()

    if len(stamps) >= 2:
        intervals = [later - earlier for earlier, later in zip(stamps, stamps[1:])]
        avg = sum(intervals) / len(intervals)
        print(
            f"采集间隔均值 {avg * 1000:.1f} ms（≈{1 / avg:.1f} FPS），"
            f"最小 {min(intervals) * 1000:.1f} / 最大 {max(intervals) * 1000:.1f} ms"
        )
    stats = source.stats.as_dict()
    print(
        f"统计: 读到 {stats['frames_read']} 帧，读失败 {stats['read_errors']} 次，"
        f"重连 {stats['reconnects']} 次，在线 {stats['uptime_s']:.2f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
