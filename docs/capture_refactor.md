# 采集抽象层与帧调度改造说明

> 版本：2026-11-04　范围：`capturekit/`（新增）、`board/`、`agent/tools.py`、`tests/`、`scripts/`、`configs/`
> 对应问题项：**C7**（无采集抽象层）、**D1**（阻塞式队列、无丢弃策略）、**C8**（摄像头当前不可用）

## 1. 改造前状态（问题复述）

| 项 | 现状 | 后果 |
|---|---|---|
| C7 | `cv2.VideoCapture` 直写在 `board/camera_infer.py`、`agent/tools.py`、`board/test_camera.py` 三处，各自设置 MJPG / 分辨率 / letterbox | 参数漂移无人察觉；换源（视频/图片/合成）必须改推理代码；无法在无摄像头环境验证链路 |
| D1 | `queue.Queue(maxsize=4)` + 阻塞 `get()`（`pipeline_infer.py:31-32`、`camera_infer.py`） | 采集线程被消费速率反压，端到端延迟 ≈ (队列长度+1) × 下游耗时，随下游变慢线性累积；延迟不可观测 |
| C8 | 板载 USB 摄像头损坏，`/dev/video*` 未枚举 | 摄像头分支代码无法实测；若无源抽象，整条流水线在无设备时直接不可运行、不可回归 |

## 2. 改造目标

1. 采集与推理解耦：上层只依赖 `FrameSource` 接口，不出现任何 `cv2.VideoCapture`；
2. 帧传递策略显式化：非阻塞、最新帧优先、陈旧帧丢弃，并带丢弃计数与帧年龄指标；
3. 无摄像头也可完整验证：合成源 / 视频源 / 图片源与摄像头源共享同一接口与同一流水线；
4. 工程约束：配置外部化 + 启动期校验、异常分类、单元测试覆盖、压测脚本可复跑。

## 3. 目标架构

```
                 ┌─────────────────────── capturekit ───────────────────────┐
配置 JSON ──► config.py（dataclass + validate，未知字段即报错）             │
                 │            │                                              │
                 │      registry.create_source()  ← 按 kind 选择实现          │
                 │            │                                              │
                 │   sources/: camera | video | image | synthetic            │
                 │            │  （均继承 base.FrameSource）                 │
                 │      base.FrameSource                                     │
                 │      ├ open()/read()/close() 三钩子 + 上下文管理器          │
                 │      ├ 序号递增、采集时间戳、连续失败阈值、异常分类          │
                 │      ├ preprocess.Preprocessor（letterbox + BGR→RGB）      │
                 │      └ stats（读帧数/失败数/uptime）                       │
                 │            │ Frame                                    │
                 │            ▼                                              │
                 │   scheduler.create_sink(policy)                           │
                 │   ├ LatestFrameSlot    非阻塞、只留最新、旧帧丢弃           │
                 │   └ BlockingFrameQueue queue.Queue(maxsize) 语义（基线）    │
                 │            │                                              │
                 │   scheduler.FramePump  采集线程：读→发布，错误按策略收敛     │
                 │            │                                              │
                 │   metrics.CaptureReport + LatencyTracker（帧年龄 p95）      │
                 └────────────┼──────────────────────────────────────────────┘
                              ▼
                    board/infer_runtime.PipelineRunner
                    （消费帧 → Detector → PipelineReport）
                              ▼
                    board/camera_infer.py CLI / agent/tools.py
```

各模块职责：

| 模块 | 对外标识 | 职责 |
|---|---|---|
| `capturekit/models.py` | `Frame`、`INPUT_SIZE` | 帧数据结构（图像 + 序号 + 时间戳），全项目统一 |
| `capturekit/errors.py` | `CaptureError` / `SourceConfigError` / `DeviceUnavailableError` / `FrameReadError` / `SourceClosedError` | 异常分类：配置错误 vs 设备缺失 vs 单帧读失败 vs 已关闭，上层可分别处理 |
| `capturekit/base.py` | `FrameSource`（ABC） | 生命周期、序号、失败阈值、预处理、统计、`preprocessor`/`is_opened` 只读属性 |
| `capturekit/sources/` | `CameraSource` / `VideoFileSource` / `ImageSource` / `SyntheticSource` | 四种源，同一接口同一语义；摄像头读失败累计达阈值转 `DeviceUnavailableError` |
| `capturekit/config.py` | `CameraConfig` / `VideoConfig` / `ImageConfig` / `SyntheticConfig` / `PreprocessConfig` / `CaptureConfig` | 配置树、`from_dict`（未知字段直接报错）、`from_file`（JSON / 可选 YAML）、`to_dict`、`summary()` |
| `capturekit/preprocess.py` | `Preprocessor`、`letterbox()` | 唯一一处 letterbox 实现，参数来自 `config.preprocess` |
| `capturekit/registry.py` | `create_source()`、`register_source()` | 源工厂；新增源只需注册，不动上层 |
| `capturekit/scheduler.py` | `LatestFrameSlot` / `BlockingFrameQueue` / `create_sink()` / `FramePump` | 两种调度语义 + 采集线程 |
| `capturekit/metrics.py` | `LatencyTracker`（avg/p95/max）、`CaptureReport` | 帧年龄与吞吐统计，可序列化落盘 |
| `capturekit/devices.py` | `probe_camera_devices()` / `has_camera()` / `absence_hint()` | 设备探测；无设备时给出 5 步排查清单而不是干瘪报错 |
| `capturekit/diagnostics.py` | `python3 -m capturekit.diagnostics` | 环境自检 + 采集冒烟入口（无需推理模型） |
| `board/infer_runtime.py` | `Detector`（Protocol）/ `VisionRuntime` / `PipelineRunner` / `PipelineReport` | 推理侧；`Detector` 为协议，测试注入假实现 |
| `scripts/bench_capture.py`、`bench_sweep.sh` | — | 调度策略对照压测，不依赖模型 / NPU / 摄像头 |

## 4. 两种调度语义

| | `latest`（默认） | `blocking`（基线） |
|---|---|---|
| 发布行为 | 非阻塞，覆盖旧帧，旧帧计入 `dropped` | 队列满则阻塞生产者 |
| 消费行为 | 取走即清空槽位 | FIFO 逐帧消费 |
| 帧新鲜度 | 恒定 ≈ 下游单帧耗时 | 随积压线性增长 |
| 丢帧 | 有（陈旧帧） | 无 |
| 适用 | 实时安全巡检：宁可丢帧也要看最新画面 | 离线复核 / 不能丢帧的统计场景 |

两者共用同一 `FramePump` 与同一消费循环，仅 sink 实现不同，切换即 `policy` 差异，代码路径不分裂。

## 5. 实测对照（开发机合成源，非板端 NPU 数字）

`scripts/bench_capture.py`，产帧 120 FPS、每档 3 s：

| 单帧处理 | 策略 | 产帧 | 消费 | 丢弃 | avg 帧年龄 | p95 帧年龄 |
|---|---|---|---|---|---|---|
| 30 ms | latest | 361 | 99 | 260 | 4.17 ms | 8.03 ms |
| 30 ms | blocking | 105 | 100 | 0 | 144.11 ms | 150.22 ms |
| 50 ms | latest | 363 | 60 | 301 | 4.49 ms | 7.89 ms |
| 50 ms | blocking | 65 | 60 | 0 | 235.47 ms | 250.13 ms |
| 100 ms | latest | 363 | 30 | 331 | 4.17 ms | 7.86 ms |
| 100 ms | blocking | 35 | 30 | 0 | 446.21 ms | 500.60 ms |

结论：`blocking` 的延迟 ≈ (队列长度+1) × 单帧耗时，与下游耗时线性相关；`latest` 的延迟稳定在一帧处理时间内，代价是丢帧数上升。

**口径声明：以上为调度语义对照实验（合成源 + 固定休眠模拟下游），与 NPU / 摄像头真实性能无关，不得与板端实测指标混用。**

## 6. 验证边界

| 项 | 状态 | 说明 |
|---|---|---|
| 单元测试 | 66 项通过 | `python3 -m pytest tests`，覆盖配置校验、四种源、调度并发不变量、Pump 异常、PipelineRunner、Agent 工具走抽象层 |
| 合成源端到端 | 通过 | `board/camera_infer.py --source synthetic`，推理后端需模型存在；开发机 `models/` 缺失时按设计返回退出码 3 并给出提示 |
| 摄像头真实链路 | **未实测** | C8：USB 摄像头损坏，`/dev/video*` 未枚举。代码路径完整，恢复设备后 `python3 board/test_camera.py` 即可验证 |
| 板端 NPU 数字 | 沿用原实测 | 本次改造未在板端复测，不产生新性能数字 |

## 7. 变更清单

新增：`capturekit/`（11 模块 + 4 源）、`board/infer_runtime.py`、`tests/`（6 文件）、`scripts/bench_capture.py`、`scripts/bench_sweep.sh`、`configs/camera_default.json`、`configs/demo_synthetic.json`、`pytest.ini`

修改：`board/camera_infer.py`（改用 capturekit + `--source/--policy/--frames`）、`board/pipeline_infer.py`（去掉手写队列）、`board/test_camera.py`（改成抽象层冒烟 + `--list`）、`agent/tools.py`（`_capture_frame` 走 `grab_frame`，异常映射为 `RuntimeError`）、`README.md`

删除：三处重复的 `cv2.VideoCapture` 直连与 `queue.Queue` 手写 pipeline、`_letterbox` 副本

## 8. 回滚

改动集中在 `capturekit/` 新增与 4 个文件的替换，`git stash` 或 `git checkout -- board/camera_infer.py board/pipeline_infer.py board/test_camera.py agent/tools.py` 即可回到改造前；`capturekit/`、`tests/`、`scripts/`、`configs/` 为纯新增，删除目录不影响原功能。
