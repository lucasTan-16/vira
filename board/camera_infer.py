#!/usr/bin/env python3
"""板端实时推理 CLI：采集 → 推理(NPU) → 后处理(CPU)。

相对于改造前的版本，本文件只做**参数解析与结果打印**，不再自己
``cv2.VideoCapture``、不再自己写 letterbox、不再直接 ``queue.Queue``：

- 采集统一走 ``capturekit`` 抽象层（C7）——摄像头/视频/图片/合成源同一接口；
- 帧调度默认 ``latest``（最新帧优先、丢弃陈旧帧、非阻塞），
  ``--policy blocking`` 可复现改造前的阻塞式行为做对照（D1）；
- 摄像头当前无设备（C8）无需改代码：``--source synthetic|video`` 走同一条流水线。

用法::

    taskset f0 python3 camera_infer.py int8                          # 摄像头实时
    taskset f0 python3 camera_infer.py int8 --source video --video clip.264 --frames 300
    taskset f0 python3 camera_infer.py int8 --source synthetic --frames 120
    taskset f0 python3 camera_infer.py int8 --policy blocking --frames 300   # 改造前基线
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "models")
if ROOT not in sys.path:      # 支持 python3 board/camera_infer.py 直接运行
    sys.path.insert(0, ROOT)

from capturekit import (
    CameraConfig,
    CaptureConfig,
    ImageConfig,
    SourceConfigError,
    SyntheticConfig,
    VideoConfig,
    absence_hint,
    probe_camera_devices,
)
from infer_runtime import PipelineReport, PipelineRunner, VisionRuntime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("camera_infer")


def build_capture_config(args: argparse.Namespace) -> CaptureConfig:
    """按命令行参数组装采集配置；给了 --config 时以配置文件为准。"""
    if args.config:
        return CaptureConfig.from_file(args.config)

    if args.source == "camera":
        return CaptureConfig(
            kind="camera",
            camera=CameraConfig(index=args.device, fps=args.fps),
        )
    if args.source == "video":
        if not args.video:
            raise SourceConfigError("--source video needs --video <path>")
        return CaptureConfig(
            kind="video",
            video=VideoConfig(path=args.video, loop=True, realtime=args.realtime),
        )
    if args.source == "image":
        if not args.image:
            raise SourceConfigError("--source image needs --image <path>")
        return CaptureConfig(kind="image", image=ImageConfig(path=args.image, loop=True))
    return CaptureConfig(
        kind="synthetic",
        synthetic=SyntheticConfig(
            width=args.width, height=args.height, fps=args.fps, pattern="gradient"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VIRA 板端实时推理：capturekit 采集 + NPU 推理 + CPU 后处理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("mode", nargs="?", default="int8", choices=["int8", "fp16"],
                        help="量化模式，决定模型文件名 best_rknnopt_<mode>.rknn")
    parser.add_argument("--model", default=None, help="直接指定 rknn 模型路径")
    parser.add_argument("--source", default="camera",
                        choices=["camera", "video", "image", "synthetic"],
                        help="采集源类型")
    parser.add_argument("--device", type=int, default=0, help="摄像头设备序号")
    parser.add_argument("--video", default=None, help="视频文件路径")
    parser.add_argument("--image", default=None, help="静态图片路径")
    parser.add_argument("--width", type=int, default=640, help="合成源宽度")
    parser.add_argument("--height", type=int, default=480, help="合成源高度")
    parser.add_argument("--fps", type=float, default=30.0, help="采集帧率（合成源/摄像头申请值）")
    parser.add_argument("--realtime", action="store_true", help="视频源按原帧率节流")
    parser.add_argument("--config", default=None, help="采集配置 JSON/YAML 路径")
    parser.add_argument("--policy", default="latest", choices=["latest", "blocking"],
                        help="帧调度策略：latest=最新帧优先（默认），blocking=改造前基线")
    parser.add_argument("--maxsize", type=int, default=4, help="blocking 队列长度")
    parser.add_argument("--frames", type=int, default=300, help="处理帧数上限")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="最长运行秒数（>0 时与 --frames 先到先停）")
    parser.add_argument("--warmup", type=int, default=5, help="预热轮数，0 表示不预热")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出统计结果")
    parser.add_argument("--quiet", action="store_true", help="只输出最终统计")
    return parser


def _warmup_frames(width: int, height: int, count: int = 2):
    """构造若干合成帧用于预热，避免首帧抖动污染统计。"""
    from capturekit import SyntheticSource

    source = SyntheticSource(width=width, height=height, fps=0, total=count)
    frames = []
    with source:
        while True:
            frame = source.read()
            if frame is None:
                break
            frames.append(frame)
    return frames


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quiet:
        logger.setLevel(logging.WARNING)

    model_path = args.model or os.path.join(MODEL_DIR, f"best_rknnopt_{args.mode}.rknn")
    try:
        config = build_capture_config(args)
        config.validate()
    except SourceConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2

    logger.info("采集配置: %s", config.summary())
    logger.info("模型: %s", model_path)
    logger.info("调度策略: %s", args.policy)

    runtime = VisionRuntime(model_path)
    try:
        runtime.load()
    except RuntimeError as exc:
        print(f"推理后端不可用: {exc}", file=sys.stderr)
        return 3

    try:
        if args.warmup > 0:
            logger.info("预热 %d 轮 ...", args.warmup)
            runtime.warmup(_warmup_frames(args.width, args.height), rounds=args.warmup)

        from capturekit import CaptureError, create_source

        try:
            source = create_source(config)
        except CaptureError as exc:
            print(f"采集源初始化失败: {exc}", file=sys.stderr)
            if config.kind == "camera":
                devices = probe_camera_devices()
                print(f"当前视频设备: {devices or '无'}", file=sys.stderr)
                print(absence_hint(), file=sys.stderr)
            return 4

        runner = PipelineRunner(
            source=source,
            detector=runtime,
            policy=args.policy,
            maxsize=args.maxsize,
        )
        report = runner.run(
            max_frames=args.frames,
            duration=args.duration if args.duration > 0 else None,
        )
    finally:
        runtime.release()

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print("=" * 62)
        print(f"模式        : {args.mode}")
        print(report.format_text())
        print("=" * 62)

    if runner.error is not None:
        print(f"运行异常终止: {runner.error!r}", file=sys.stderr)
        print("排查建议: ls /dev/video* ；v4l2-ctl --list-devices ；"
              "无设备时改用 --source synthetic|video", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
