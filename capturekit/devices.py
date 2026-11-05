"""摄像头设备探测：把"没有设备"这一类现场问题变成可读诊断。

板端常见故障是设备根本没枚举出来（USB 掉线、供电不足、hub 不识别），此时
上层只看到一句 "opened: False" 很难排查。本模块直接读 ``/dev/video*`` 与
sysfs 名称，供 CLI、冒烟脚本、诊断工具共用，避免每处各写一遍。
"""
from __future__ import annotations

import glob
import os

VIDEO_GLOB = "/dev/video*"
V4L_SYSFS = "/sys/class/video4linux"


def _device_index(path: str) -> int:
    """从 ``/dev/video12`` 抽出 12，用于稳定排序。"""
    digits = "".join(ch for ch in os.path.basename(path) if ch.isdigit())
    return int(digits) if digits else -1


def _sysfs_name(node: str) -> str:
    """读 ``/sys/class/video4linux/videoX/name``；读不到返回空串。"""
    try:
        with open(os.path.join(V4L_SYSFS, node, "name"), encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def probe_camera_devices() -> list[str]:
    """返回已枚举的视频设备描述列表，如 ``['/dev/video0 (USB Camera: USB Camera)']``。

    非 Linux 平台或未挂载 devtmpfs 时返回空列表，不抛异常。
    """
    devices: list[str] = []
    for path in sorted(glob.glob(VIDEO_GLOB), key=_device_index):
        node = os.path.basename(path)
        name = _sysfs_name(node)
        devices.append(f"{path} ({name})" if name else path)
    return devices


def has_camera() -> bool:
    """是否存在任何 ``/dev/video*`` 节点。"""
    return bool(glob.glob(VIDEO_GLOB))


def absence_hint() -> str:
    """没有设备时给出排查清单，供 CLI 直接打印。"""
    return (
        "未检测到 /dev/video* 设备，排查顺序：\n"
        "  1) lsusb 是否看到摄像头（看不到 = 供电/线缆/USB 口问题）\n"
        "  2) dmesg | tail -30 是否有 uvcvideo 枚举失败日志\n"
        "  3) ls /dev/video* 是否存在节点（有 USB 但无节点 = 驱动未绑定）\n"
        "  4) 设备是否被其他进程独占（fuser -v /dev/video0）\n"
        "  5) 无摄像头时可用 --source synthetic / video 走同一条流水线验证逻辑"
    )
