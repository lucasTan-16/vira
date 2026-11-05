# VIRA — 基于 RK3588s 的端侧多模态巡检 Agent

**Vision → Inference → Reasoning → Action**

一个在 RK3588 NPU 上跑通「感知 → 量化推理 → 合规判定 → 语言决策 → 硬件动作」完整闭环的端侧多模态 Agent。USB 摄像头采集画面，NPU 上的 YOLO 做 PPE 检测，确定性规则层做合规判定，小语言模型通过 function calling 做自然语言决策与工具编排，最终经 GPIO 驱动真实硬件完成报警/停机；**全程本地运行，不依赖云端**。

```
USB 摄像头 (UVC, MJPG 640x480)
      │ capturekit 采集抽象层（V4L2 参数下发 + letterbox + 最新帧优先调度）
      ▼
[V] YOLOv11n (RKNN INT8, NPU Core0) ── 6 类实体框（Person + 5 类 PPE）
      ▼
[C] PPE 合规规则层（确定性 Python）── 人-装备空间匹配 (IoU/区域归属)
      ▼
[R] Qwen2.5-0.5B (RKLLM W8A8, NPU 三核) ── 手写 Agent Loop + function calling
      │        run_detection / query_state / control_device
      ▼
[A] GPIO：继电器 / LED ── 报警 / 停机 / 指示，结果回传自然语言回复
```

## 核心特性

- **采集抽象层与帧调度解耦**：摄像头/视频/图片/合成源统一 `FrameSource` 接口，参数下发、预处理、异常分类、统计收敛到一处；帧传递默认「最新帧优先 + 丢弃陈旧帧 + 非阻塞」，不再让队列长度隐式决定端到端延迟（见 [采集抽象层与帧调度](#采集抽象层与帧调度capturekit)）；
- **三层解耦**：视觉只做实体检测，确定性规则层做可审计的合规判定，LLM 只做决策编排与自然语言，不做精确空间几何；
- **双模型同板驻留**：视觉 RKNN 与语言 RKLLM 在 4GB 板上串行分时调度，NPU Core0 分时复用，无 OOM、无资源争抢；
- **手写 Agent，不依赖框架**：基于 asyncio 手写工具注册表、function calling 解析与迭代兜底，每个环节可解释、可追问；
- **小模型可靠性工程**：工具返回确定性 `summary`，模型只转述结论，从机制上消除 0.5B 对数字的误读。

## 实测指标

> 全部为 RK3588 真机实测，非估算。

**视觉（YOLOv11n，640×640）**

| 指标 | FP32 (PC) | FP16 (NPU) | INT8 (NPU) |
|---|---|---|---|
| mAP50 | 0.817 | 0.8131 | **0.811**（板端实测，141 图 / 1097 GT） |
| mAP50-95 | 0.447 | 0.4432 | 0.444 |
| 单帧纯 NPU 延迟 | — | 48.7 ms | **24.1 ms** |
| 单帧帧率 | — | 20.5 FPS | 41.5 FPS |
| 模型体积 | 5.3 MB | 7.3 MB | **4.7 MB** |
| 单实例峰值 RSS | — | 122.8 MB | 103.4 MB |

**调度与吞吐**

| 场景 | 结果 |
|---|---|
| 端到端串行（NPU 24.3 + 后处理 11.6） | 35.9 ms |
| 流水线（后处理被掩盖） | 34.4 FPS |
| 流水线 + taskset 绑 A76 大核 | **41.7 FPS（较默认 +17.5%）** |
| 摄像头全链路实时（MJPG 640x480） | 采集为瓶颈段，当前不可复测（板端无 `/dev/video*`，见 §采集抽象层） |
| 多实例三核聚合吞吐 | **97.0 FPS**（三实例分核并发聚合） |

**语言模型（Qwen2.5-0.5B-Instruct，RKLLM W8A8）**

| 指标 | 数值 |
|---|---|
| decode 速度 | **24.2 tok/s**（457 token 端到端） |
| 首 token 延迟 TTFT | 192 ms（42 token prefill） |
| 峰值内存 | 683.7 MB（模型文件 757 MB） |

**合规规则层 / 系统**

| 指标 | 数值 |
|---|---|
| 规则判定一致率（vs 原始标注，616 判定） | **87.5%**，Cohen Kappa 0.706 |
| 双模型驻留内存增量 | 704 MB（LLM 677 + 视觉 27），加载后仍余约 2.3 GB |

## 采集抽象层与帧调度（capturekit）

改造前的问题：`cv2.VideoCapture` 散落在 `board/camera_infer.py`、`agent/tools.py` 与冒烟脚本三处，
各自设置 MJPG/分辨率/letterbox；帧传递用 `queue.Queue(maxsize=4)` + 阻塞 `get()`，
采集线程会被消费速率反压——下游一慢，画面延迟随积压线性增长，且这个延迟是隐式的、没有指标可查。

改造后的分层：

| 层 | 模块 | 职责 |
|---|---|---|
| 抽象接口 | `capturekit/base.py` | `FrameSource` 三个钩子（open/read/close），统一序号、采集时间戳、预处理、统计、异常分类、失败重连 |
| 具体源 | `capturekit/sources/` | camera（V4L2/UVC）、video、image、synthetic 四种，同一接口同一语义 |
| 配置 | `capturekit/config.py` | dataclass 配置树，启动期校验（非法分辨率/格式直接报错，不带病运行），支持 dict/JSON/YAML |
| 预处理 | `capturekit/preprocess.py` | letterbox 640×640 + BGR→RGB，全项目唯一实现 |
| 帧调度 | `capturekit/scheduler.py` | `LatestFrameSlot`（非阻塞、只保留最新帧）、`BlockingFrameQueue`（改造前基线）、`FramePump` 采集线程 |
| 指标 | `capturekit/metrics.py` | 采集/推理/后处理吞吐、丢弃帧数、端到端帧年龄 avg/p95/max |
| 设备探测 | `capturekit/devices.py` | `/dev/video*` + sysfs 名称，无设备时直接给排查清单 |

两种调度语义（可用 `--policy` 切换，同一份代码路径）：

- `latest`（默认）：采集线程永不阻塞，槽位只存最新一帧，消费者取走即清空；来不及消费的旧帧计数丢弃。
  消费者拿到的**永远是最新画面**，端到端延迟与下游快慢无关。
- `blocking`：等价改造前的 `queue.Queue(maxsize=N)`，用于回归对照与完整性优先场景。

### 调度策略对照（开发机合成源，非板端 NPU 数字）

产帧 120 FPS，每档 3 s，`scripts/bench_capture.py` 可复跑：

| 单帧处理耗时 | 策略 | 产帧 | 消费 | 丢弃 | avg 帧年龄 | p95 帧年龄 |
|---|---|---|---|---|---|---|
| 30 ms | latest | 361 | 99 | 260 | 4.17 ms | 8.03 ms |
| 30 ms | blocking | 105 | 100 | 0 | 144.11 ms | 150.22 ms |
| 50 ms | latest | 363 | 60 | 301 | 4.49 ms | 7.89 ms |
| 50 ms | blocking | 65 | 60 | 0 | 235.47 ms | 250.13 ms |
| 100 ms | latest | 363 | 30 | 331 | 4.17 ms | 7.86 ms |
| 100 ms | blocking | 35 | 30 | 0 | 446.21 ms | 500.60 ms |

读法：blocking 的端到端延迟 ≈ (队列长度+1) × 单帧处理耗时，下游变慢就线性累积；latest 的延迟恒定在
单帧处理时间内，代价是丢弃陈旧帧。**"要新鲜度还是要完整性"变成显式配置项，不再由队列长度被动决定。**

> 说明：上表是调度语义的对照实验（合成源 + 固定休眠模拟下游），与摄像头、NPU 真实性能无关，
> 不可与 §实测指标 中的板端数字混用。

### 用法

```bash
# 摄像头实时（当前 USB 摄像头损坏，见下方说明）
python3 board/camera_infer.py int8

# 无摄像头时走同一条流水线：视频文件 / 合成源
python3 board/camera_infer.py int8 --source video --video data/clip.mp4 --frames 300
python3 board/camera_infer.py int8 --source synthetic --frames 120

# 改造前基线对照
python3 board/camera_infer.py int8 --policy blocking --frames 300

# 采集链路冒烟 + 设备探测
python3 board/test_camera.py --list
python3 board/test_camera.py --frames 10

# 环境自检 + 调度语义自检（不需要模型 / NPU / 摄像头）
python3 -m capturekit.diagnostics --selftest

# 调度策略压测（不需要模型/NPU/摄像头）
python3 scripts/bench_capture.py --consumer-ms 50 --duration 3
bash scripts/bench_sweep.sh

# 单元测试（66 项，无摄像头与 NPU 可跑）
python3 -m pytest tests -q
```

改造细节（分层、调度语义、对照数据、验证边界、回滚方式）见 [docs/capture_refactor.md](docs/capture_refactor.md)。

> **摄像头现状**：板载 USB 摄像头当前已损坏（`/dev/video*` 未枚举），摄像头分支的代码路径保留完整，
> 但本仓库内的调度与流水线验证全部走合成源 / 视频源，**未在真实摄像头上复测**。
> 恢复设备后 `python3 board/test_camera.py` 即可确认链路。

## 目录结构

```
vira/
├── capturekit/  采集抽象层：FrameSource 接口、四种源、配置校验、预处理、帧调度、指标、设备探测
├── vision/      数据清洗、YOLOv11n 训练、ONNX 导出、RKNN 转换与校准
├── board/       RK3588 板端推理流水线、实时 CLI、摄像头冒烟、内存测量
├── compliance/  确定性 PPE 合规规则层 + 单元测试 + 一致率验证
├── llm/         RKLLM 转换脚本（W8A8/FP16）、C API 头文件与示例
├── agent/       手写 Agent Loop、RKLLM 运行时、工具注册表（采集经 capturekit）、CLI
├── scripts/     采集调度压测（bench_capture / bench_sweep）
├── tests/       capturekit 与流水线单元测试（无需摄像头与 NPU）
└── eval/        精度 / 延迟 / 内存 benchmark 与口径对齐
```

## 技术栈

- **语言/绑定**：Python 3.10、ctypes（C API 绑定）、NumPy
- **训练/转换**：PyTorch、ultralytics、RKNN-Toolkit2、RKLLM-Toolkit
- **板端运行时**：RKNNLite、librkllmrt、OpenCV/V4L2、libgpiod
- **Agent**：asyncio（手写，不依赖 LangChain）
- **测试**：pytest（66 项，覆盖配置 / 四种源 / 调度并发不变量 / 流水线，无摄像头与 NPU 可跑）

## 硬件

- 板端：香橙派 5（RK3588S，4GB，三核 NPU 6 TOPS）
- 训练/转换：PC（RTX 2070 8GB）
- 外设：USB UVC 摄像头、继电器 / LED

## 路线图

- [x] 6 类 PPE 视觉训练，test mAP50 0.817
- [x] RKNN INT8 量化上板（mAP50 0.8131 → 0.811，−0.21 pp）与多核调度
- [x] 合规规则层 + Qwen2.5-0.5B 上板 + 手写 Agent 闭环
- [ ] 双模型调度、GPIO 真机动作、全链路联调
- [x] 采集抽象层（capturekit）+ 非阻塞 / 最新帧优先 / 丢弃陈旧帧调度，66 项单测覆盖
- [ ] 摄像头设备恢复后在真实 UVC 上复测采集链路
- [ ] VLA 仿真与端侧可行性评估（可选拓展）
