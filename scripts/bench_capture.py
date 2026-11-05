#!/usr/bin/env python3
"""帧调度策略对照压测：latest（最新帧优先） vs blocking（改造前基线）。

不需要模型、NPU、摄像头——用合成源按指定速率产帧，用固定休眠模拟下游处理耗时，
直接对比两种调度策略下的：

- 采集 FPS：慢消费时采集能力是否被反压；
- 端到端帧年龄（采集时刻 → 处理完成）：画面新鲜度；
- 丢弃帧数：能量账目是否对得上。

用法::

    python3 scripts/bench_capture.py                      # 默认参数，两种策略各跑一次
    python3 scripts/bench_capture.py --duration 5 --producer-fps 120 --consumer-ms 80
    python3 scripts/bench_capture.py --json-out runs/capture_bench.json
    python3 scripts/bench_capture.py --policies blocking   # 只跑基线
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from capturekit import (  # noqa: E402
    FramePump,
    LatencyTracker,
    SyntheticSource,
    create_sink,
)


@dataclass
class BenchRow:
    policy: str
    window_s: float
    produced: int
    consumed: int
    dropped: int
    producer_fps: float
    consumer_fps: float
    avg_ms: float
    p95_ms: float
    max_ms: float


def run_case(
    policy: str,
    producer_fps: float,
    consumer_ms: float,
    duration: float,
    maxsize: int,
    width: int,
    height: int,
) -> BenchRow:
    source = SyntheticSource(width=width, height=height, fps=producer_fps)
    sink = create_sink(policy, maxsize=maxsize)          # type: ignore[arg-type]
    pump = FramePump(source, sink)
    latency = LatencyTracker()
    consumed = 0

    pump.start()
    started = time.perf_counter()
    try:
        while time.perf_counter() - started < duration:
            frame = sink.get(timeout=0.2)
            if frame is None:
                break
            latency.record(frame.age_ms())
            consumed += 1
            time.sleep(consumer_ms / 1000.0)
    finally:
        sink.close()                                     # 先解阻塞投递
        pump.stop(timeout=3.0)
    window = time.perf_counter() - started

    snap = latency.snapshot()
    stats = sink.stats()
    return BenchRow(
        policy=policy,
        window_s=round(window, 3),
        produced=source.stats.frames_read,
        consumed=consumed,
        dropped=stats.dropped,
        producer_fps=round(source.stats.frames_read / window, 2) if window else 0.0,
        consumer_fps=round(consumed / window, 2) if window else 0.0,
        avg_ms=snap["avg_ms"],
        p95_ms=snap["p95_ms"],
        max_ms=snap["max_ms"],
    )


def print_table(rows: list[BenchRow]) -> None:
    header = ("策略", "产帧", "消费", "丢弃", "产帧FPS", "消费FPS", "avg年龄ms", "p95年龄ms")
    widths = (10, 8, 8, 8, 10, 10, 12, 12)
    print("".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-" * sum(widths))
    for r in rows:
        cells = (
            r.policy,
            str(r.produced),
            str(r.consumed),
            str(r.dropped),
            f"{r.producer_fps:.1f}",
            f"{r.consumer_fps:.1f}",
            f"{r.avg_ms:.2f}",
            f"{r.p95_ms:.2f}",
        )
        print("".join(c.ljust(w) for c, w in zip(cells, widths)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="帧调度策略对照压测")
    parser.add_argument("--policies", nargs="+", default=["latest", "blocking"],
                        choices=["latest", "blocking"])
    parser.add_argument("--producer-fps", type=float, default=120.0, help="采集速率（<=0 不限速）")
    parser.add_argument("--consumer-ms", type=float, default=50.0, help="模拟单帧处理耗时")
    parser.add_argument("--duration", type=float, default=3.0, help="每个策略的运行秒数")
    parser.add_argument("--maxsize", type=int, default=4, help="blocking 队列长度")
    parser.add_argument("--width", type=int, default=640, help="合成帧宽")
    parser.add_argument("--height", type=int, default=480, help="合成帧高")
    parser.add_argument("--json-out", default=None, help="把结果写入 JSON 文件")
    args = parser.parse_args(argv)

    print(
        f"参数: 产帧 {args.producer_fps} FPS / 单帧处理 {args.consumer_ms} ms / "
        f"{args.duration} s / 帧尺寸 {args.width}x{args.height} / blocking 队列 {args.maxsize}\n"
    )
    rows = [
        run_case(
            policy=p,
            producer_fps=args.producer_fps,
            consumer_ms=args.consumer_ms,
            duration=args.duration,
            maxsize=args.maxsize,
            width=args.width,
            height=args.height,
        )
        for p in args.policies
    ]
    print_table(rows)

    payload = {
        "params": {
            "producer_fps": args.producer_fps,
            "consumer_ms": args.consumer_ms,
            "duration_s": args.duration,
            "maxsize": args.maxsize,
            "frame": f"{args.width}x{args.height}",
        },
        "rows": [asdict(r) for r in rows],
    }
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\n结果已写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
