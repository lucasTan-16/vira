"""USB/UVC 摄像头采集源（V4L2）。

对应板端 ``/dev/videoN``。当前板子上的 USB 摄像头硬件已损坏（设备节点缺失），
本模块的职责是：设备在时稳定采帧，设备不在时**快速失败并给出可执行的排查信息**，
而不是让上层收到一个含义不明的 ``cap.read() is False``。

关键工程点：
1. MJPG 而非默认 YUYV——YUYV 传 640x480 会把带宽/帧率拖死；
2. ``CAP_PROP_BUFFERSIZE=1``——把驱动侧队列压到最小，配合上层 latest-frame 策略，
   避免"取到的永远是历史帧"；设置是否生效会记录到日志，不静默忽略；
3. 连续读失败升级为 :class:`DeviceUnavailableError`，避免采集线程空转；
4. 提供 :meth:`CameraSource.probe` 供诊断无副作用地探活。
"""
from __future__ import annotations

import logging
import os
from typing import Any

import cv2

from ..base import FrameSource
from ..config import CameraConfig
from ..errors import DeviceUnavailableError, FrameReadError
from ..preprocess import Preprocessor

logger = logging.getLogger(__name__)


class CameraSource(FrameSource):
    """V4L2 摄像头源。

    Args:
        config: 摄像头参数（设备号、分辨率、FOURCC、缓冲队列长度等）。
        preprocessor: 预处理对象，缺省 letterbox 640x640 + RGB。
    """

    kind = "camera"

    def __init__(
        self,
        config: CameraConfig | None = None,
        preprocessor: Preprocessor | None = None,
    ) -> None:
        self.config = config or CameraConfig()
        super().__init__(
            name=f"camera:{self.config.index}",
            preprocessor=preprocessor,
            max_read_failures=self.config.max_read_failures,
        )
        self._cap: cv2.VideoCapture | None = None

    # ===== 打开 =====
    def _open_impl(self) -> None:
        device = f"/dev/video{self.config.index}"
        cap = cv2.VideoCapture(self.config.index, self._backend_id())
        if not cap.isOpened():
            cap.release()
            raise DeviceUnavailableError(
                f"cannot open camera {self.config.index} ({device}): "
                f"device missing, busy or permission denied. "
                f"check: ls -l {device}; v4l2-ctl --list-devices; "
                f"groups | grep video"
            )
        self._cap = cap
        self._apply_properties(cap)

    def _backend_id(self) -> int:
        backend = (self.config.backend or "v4l2").lower()
        mapping = {
            "v4l2": getattr(cv2, "CAP_V4L2", 0),
            "any": getattr(cv2, "CAP_ANY", 0),
        }
        if backend not in mapping:
            raise DeviceUnavailableError(f"unsupported camera backend: {self.config.backend}")
        return mapping[backend]

    def _apply_properties(self, cap: cv2.VideoCapture) -> None:
        """下发采集参数，并核对是否真正生效（驱动可能静默拒绝）。"""
        cfg = self.config
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        # 驱动侧队列压到最小：配合最新帧优先策略，减少历史帧积压
        buffers_ok = cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.buffer_size)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (cfg.width, cfg.height):
            logger.warning(
                "%s: resolution %dx%d not applied, driver reports %dx%d",
                self.name, cfg.width, cfg.height, actual_w, actual_h,
            )
        if not buffers_ok:
            logger.warning(
                "%s: CAP_PROP_BUFFERSIZE=%d not supported by driver; "
                "stale frames will rely on latest-frame policy only",
                self.name, cfg.buffer_size,
            )

    # ===== 读取 =====
    def _read_impl(self):
        if self._cap is None:
            raise DeviceUnavailableError(f"{self.name}: capture not initialised")
        ok, raw = self._cap.read()
        if not ok or raw is None:
            reason = "cv2.VideoCapture.read() returned (False, None)"
            self._record_failure(reason)          # 达上限时此处直接抛 DeviceUnavailableError
            raise FrameReadError(f"{self.name}: {reason}")
        return raw

    def _close_impl(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ===== 诊断 =====
    @classmethod
    def probe(cls, config: CameraConfig | None = None) -> dict[str, Any]:
        """无副作用探活：打开、读配置回读、可选读一帧，然后释放。

        Returns:
            诊断字典：``available`` / ``device_exists`` / ``detail`` /
            ``resolution`` / ``fourcc_ok``。
        """
        cfg = config or CameraConfig()
        device = f"/dev/video{cfg.index}"
        result: dict[str, Any] = {
            "device": device,
            "device_exists": os.path.exists(device),
            "available": False,
            "detail": "",
        }
        cap = cv2.VideoCapture(cfg.index, getattr(cv2, "CAP_V4L2", 0))
        try:
            if not cap.isOpened():
                result["detail"] = "device node missing, busy or permission denied"
                return result
            result["resolution"] = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            ok, frame = cap.read()
            result["available"] = bool(ok and frame is not None)
            result["detail"] = "ok" if result["available"] else "opened but no frame (check cable/USB power)"
            return result
        finally:
            cap.release()
