"""视频文件采集源：用于无摄像头环境下的全链路回归与压测。

与摄像头源的差别：
- 帧是"要多少有多少"的确定性数据，便于复现问题；
- 可选择按文件原始帧率节流（``realtime=True``），模拟真实采集节拍；
- 可选择循环播放（``loop=True``），配合压力测试长时间跑。
"""
from __future__ import annotations

import logging
import os
import time

import cv2

from ..base import FrameSource
from ..config import VideoConfig
from ..errors import DeviceUnavailableError, FrameReadError
from ..preprocess import Preprocessor

logger = logging.getLogger(__name__)


class VideoFileSource(FrameSource):
    """本地视频文件源（mp4/avi/mkv，OpenCV 可解码即可）。"""

    kind = "video"

    def __init__(
        self,
        config: VideoConfig,
        preprocessor: Preprocessor | None = None,
    ) -> None:
        self.config = config
        super().__init__(
            name=f"video:{os.path.basename(config.path)}",
            preprocessor=preprocessor,
            max_read_failures=config.max_read_failures,
        )
        self._cap: cv2.VideoCapture | None = None
        self._next_deadline = 0.0
        self._frame_interval = 0.0

    def _open_impl(self) -> None:
        path = self.config.path
        if not path:
            raise DeviceUnavailableError("video source: empty path")
        if not os.path.exists(path):
            raise DeviceUnavailableError(f"video source: file not found: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            raise DeviceUnavailableError(f"video source: cannot open/decode: {path}")
        self._cap = cap
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.config.realtime and fps > 1.0:
            self._frame_interval = 1.0 / fps
            self._next_deadline = time.perf_counter()
        logger.info(
            "%s: %d frames @ %.1f fps, loop=%s, realtime=%s",
            self.name, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), fps,
            self.config.loop, self.config.realtime,
        )

    def _read_impl(self):
        if self._cap is None:
            raise DeviceUnavailableError(f"{self.name}: capture not initialised")

        self._pace()

        ok, raw = self._cap.read()
        if not ok or raw is None:
            if self.config.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, raw = self._cap.read()
                if ok and raw is not None:
                    return raw
            return None
        return raw

    def _pace(self) -> None:
        """按原始帧率节流，模拟真实采集到达节奏。"""
        if self._frame_interval <= 0:
            return
        now = time.perf_counter()
        if now < self._next_deadline:
            time.sleep(self._next_deadline - now)
        self._next_deadline = max(now, self._next_deadline) + self._frame_interval

    def _close_impl(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ImageSource(FrameSource):
    """静态图像源：一张图重复产出，用于接口联调与规则层验证。"""

    kind = "image"

    def __init__(
        self,
        path: str,
        loop: bool = True,
        preprocessor: Preprocessor | None = None,
        max_read_failures: int = 5,
    ) -> None:
        self.path = path
        self.loop = loop
        super().__init__(
            name=f"image:{os.path.basename(path)}",
            preprocessor=preprocessor,
            max_read_failures=max_read_failures,
        )
        self._image = None

    def _open_impl(self) -> None:
        if not self.path or not os.path.exists(self.path):
            raise DeviceUnavailableError(f"image source: file not found: {self.path}")
        image = cv2.imread(self.path)
        if image is None:
            raise DeviceUnavailableError(f"image source: cannot decode image: {self.path}")
        self._image = image

    def _read_impl(self):
        if self._image is None:
            raise FrameReadError(f"{self.name}: image not loaded")
        if self.stats.frames_read > 0 and not self.loop:
            return None
        return self._image

    def _close_impl(self) -> None:
        self._image = None
