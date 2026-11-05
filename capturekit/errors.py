"""采集层异常：按"可恢复 / 不可恢复"分类，供上层统一处理。

- CaptureError             采集层基类
- SourceConfigError        配置非法（启动期，不可恢复）
- DeviceUnavailableError   设备不存在/被占用/打开失败（不可恢复，需人工介入）
- FrameReadError           单帧读取失败（可重试，超过阈值升级为不可恢复）
- SourceClosedError        对已关闭的源继续读取（调用方逻辑错误）
"""
from __future__ import annotations


class CaptureError(Exception):
    """采集层所有异常的基类。"""


class SourceConfigError(CaptureError):
    """采集源配置非法：字段缺失、取值越界、类型不符。"""


class DeviceUnavailableError(CaptureError):
    """采集设备不可用：不存在、被占用、无权限、驱动未加载。

    该异常代表不可自愈的故障，调用方应向上暴露给用户/运维，而不是无限重试。
    """


class FrameReadError(CaptureError):
    """单帧读取失败，通常是瞬时的（USB 抖动、曝光未就绪、解码失败）。

    调用方可重试；连续超过 ``max_read_failures`` 后应由采集线程升级为故障。
    """


class SourceClosedError(CaptureError):
    """对已关闭的采集源继续调用 read/open。"""
