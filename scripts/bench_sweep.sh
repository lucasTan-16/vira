#!/usr/bin/env bash
# 扫描不同下游处理耗时下的调度策略对照，落盘 JSON 便于写材料时引用。
#
# 用法:
#   bash scripts/bench_sweep.sh                 # 默认 30/50/100 ms
#   CONSUMERS="20 40" DURATION=5 bash scripts/bench_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DURATION=${DURATION:-3}
CONSUMERS=${CONSUMERS:-"30 50 100"}

echo "扫描下游处理耗时: ${CONSUMERS} ms，每档 ${DURATION}s"
for c in $CONSUMERS; do
  echo "===== 下游处理 ${c} ms ====="
  python3 scripts/bench_capture.py \
      --consumer-ms "$c" \
      --duration "$DURATION" \
      --json-out "runs/capture_bench_c${c}ms.json" | sed -n '4,7p'
  echo
done
