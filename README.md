# VIRA — 端侧视觉-推理-行动巡检智能体

**Vision → Inference → Reasoning → Action**

在 RK3588 NPU 上跑通"感知 → 量化推理 → 语言决策 → 硬件动作"闭环的端侧多模态 Agent。摄像头看到目标，NPU 上的 YOLO 做检测，小语言模型做自然语言决策并调用工具，最终通过 GPIO 控制真实硬件完成报警/停机动作；全程不依赖云端。

## 架构

| 阶段 | 模块 | 技术 |
|---|---|---|
| **V**ision | 视觉感知 | USB 摄像头 + YOLOv11n |
| **I**nference | 端侧推理 | RKNN INT8 视觉 + RKLLM W4A16 语言（NPU） |
| **R**easoning | 语言决策 | Qwen2.5-0.5B function calling（手写 Agent） |
| **A**ction | 动作闭环 | GPIO 继电器/LED/电机 |

## 目录

- `vision/`：模型训练、ONNX 导出、RKNN 转换
- `board/`：RK3588 板端服务（视觉 / 语言 / GPIO）
- `agent/`：Agent Loop、工具注册、记忆
- `eval/`：精度 / 延迟 / 内存 benchmark 与权衡表
- `data/`、`configs/`、`tests/`、`docs/`

## 硬件

- 香橙派 5（RK3588S，4GB，NPU 6 TOPS 三核）
- PC：RTX 2070 8GB（训练 / 转换）
- USB 摄像头、继电器 / LED / 电机

## 实测指标

> 全部真机实测，开发中，待填。

| 指标 | 数值 |
|---|---|
| 视觉 mAP50-95 | 待实测 |
| 视觉 INT8 单帧延迟 | 待实测 |
| 语言 decode tokens/s | 待实测 |
| 端到端响应 | 待实测 |
| 双模型同板内存峰值 | 待实测 |

## 进度

- [x] Week 1：视觉训练（进行中）
- [ ] Week 2：视觉上板 + 量化
- [ ] Week 3：语言上板 + Agent
- [ ] Week 4：双模型闭环 + 硬件动作