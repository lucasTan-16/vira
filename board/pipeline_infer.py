#!/usr/bin/env python3
"""板端连续帧率（流水线）验证：NPU 推理与 CPU 后处理并行。

改造后本脚本只负责"造固定帧 + 打印吞吐"，三线程流水线由
``infer_runtime.PipelineRunner`` 统一实现：

- 源由 ``capturekit`` 合成源提供（同一帧重复，等价于原来的 sample 重复投递），
  不再手写 ``queue.Queue`` + put 循环；
- 帧调度默认 ``latest``；``--policy blocking --maxsize 4`` 可复现改造前的阻塞式口径，
  用于对比"消费慢时采集是否被反压"；
- 稳态吞吐 = 后处理完成速率（= max(NPU, 后处理)），由 ``post_fps`` 给出。

用法::

    taskset f0 python3 pipeline_infer.py int8
    taskset f0 python3 pipeline_infer.py int8 --frames 300 --policy blocking
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "models")
SAMPLE = os.path.join(HERE, "sample_640.npy")
if ROOT not in sys.path:      # 支持 python3 board/pipeline_infer.py 直接运行
    sys.path.insert(0, ROOT)

from capturekit import SyntheticSource
from infer_runtime import PipelineRunner, VisionRuntime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline_infer")


def build_static_source(sample_path: str = SAMPLE, pattern: str = "noise") -> SyntheticSource:
    """构造"内容恒定"的合成源：每帧都是同一张图，只测吞吐不测画质。

    ``sample_640.npy`` 为 640x640x3 RGB；合成源约定输入 BGR，故先转 BGR，
    预处理层再转回 RGB，最终送入 NPU 的张量与改造前完全一致。
    """
    if os.path.exists(sample_path):
        image = np.load(sample_path)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"unexpected sample shape: {image.shape}")
        frame_bgr = image[..., ::-1].copy()          # RGB -> BGR
        height, width = frame_bgr.shape[:2]

        def factory(seq: int, shape: tuple[int, int]) -> np.ndarray:
            return frame_bgr

        logger.info("固定样本源: %s (%dx%d)", os.path.basename(sample_path), width, height)
        return SyntheticSource(width=width, height=height, fps=0, frame_factory=factory)

    logger.warning("样本 %s 不存在，退回合成图案 pattern=%s", sample_path, pattern)
    return SyntheticSource(width=640, height=480, fps=0, pattern=pattern)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VIRA 板端连续吞吐验证（NPU + 后处理并行）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("mode", nargs="?", default="int8", choices=["int8", "fp16"])
    parser.add_argument("--model", default=None, help="直接指定 rknn 模型路径")
    parser.add_argument("--sample", default=SAMPLE, help="固定样本 npy 路径")
    parser.add_argument("--frames", type=int, default=300, help="处理帧数上限")
    parser.add_argument("--duration", type=float, default=0.0, help="最长运行秒数（>0 生效）")
    parser.add_argument("--policy", default="latest", choices=["latest", "blocking"],
                        help="帧调度策略：latest=最新帧优先，blocking=改造前基线")
    parser.add_argument("--maxsize", type=int, default=4, help="blocking 队列长度")
    parser.add_argument("--warmup", type=int, default=10, help="预热轮数")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出统计结果")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_path = args.model or os.path.join(MODEL_DIR, f"best_rknnopt_{args.mode}.rknn")

    runtime = VisionRuntime(model_path)
    try:
        runtime.load()
    except RuntimeError as exc:
        print(f"推理后端不可用: {exc}", file=sys.stderr)
        return 3

    try:
        source = build_static_source(args.sample)
        if args.warmup > 0:
            logger.info("预热 %d 轮 ...", args.warmup)
            with SyntheticSource(width=640, height=480, fps=0, total=1) as warm:
                frame = warm.read()
            if frame is not None:
                runtime.warmup([frame], rounds=args.warmup)

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
        fps = report.post_fps
        ms = (1000.0 / fps) if fps > 0 else float("inf")
        print("=" * 62)
        print(f"模式 {args.mode}，流水线 {report.frames_postprocessed} 帧")
        print(f"并行墙钟: {report.wall_s:.3f} s")
        print(f"连续吞吐: {fps:.1f} FPS ({ms:.3f} ms/帧)")
        print(report.format_text())
        print("=" * 62)

    if runner.error is not None:
        print(f"流水线异常终止: {runner.error!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
