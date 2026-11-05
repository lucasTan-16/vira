"""采集源工厂：按配置构造具体 FrameSource，新增源类型只需在此注册。

上层（推理脚本、Agent 工具、CLI）只跟 :func:`create_source` 打交道，
不再直接 import 具体实现，也不出现 ``cv2.VideoCapture`` 等硬件细节。
"""
from __future__ import annotations

import logging
from typing import Callable

from .base import FrameSource
from .config import CaptureConfig
from .errors import SourceConfigError
from .models import Frame
from .preprocess import Preprocessor
from .sources.camera import CameraSource
from .sources.synthetic import SyntheticSource
from .sources.video import ImageSource, VideoFileSource

logger = logging.getLogger(__name__)

Builder = Callable[[CaptureConfig, Preprocessor | None], FrameSource]

_BUILDERS: dict[str, Builder] = {}


def register_source(kind: str, builder: Builder) -> None:
    """注册一种采集源类型。重复注册会覆盖（便于测试注入替身）。"""
    _BUILDERS[kind] = builder
    logger.debug("source kind registered: %s", kind)


def supported_kinds() -> tuple[str, ...]:
    """已注册的源类型列表。"""
    return tuple(sorted(_BUILDERS))


def create_source(
    config: CaptureConfig | None = None,
    preprocessor: Preprocessor | None = None,
) -> FrameSource:
    """按配置创建（未打开的）采集源。

    Raises:
        SourceConfigError: 配置非法或类型未注册。
    """
    cfg = config or CaptureConfig()
    cfg.validate()
    builder = _BUILDERS.get(cfg.kind)
    if builder is None:
        raise SourceConfigError(
            f"capture kind not registered: {cfg.kind!r}; available {supported_kinds()}"
        )
    if preprocessor is None:
        # 配置里的预处理参数统一在此生效，各源不再自行 new Preprocessor
        preprocessor = Preprocessor(cfg.preprocess)
    source = builder(cfg, preprocessor)
    logger.info("source created: %s (%s)", source.name, cfg.summary())
    return source


def grab_frame(config: CaptureConfig | None = None) -> Frame:
    """一次性取一帧（打开→读→关闭），供 Agent 工具等单次调用场景使用。

    异常语义与 :class:`~capturekit.base.FrameSource` 保持一致：
    设备不可用抛 :class:`DeviceUnavailableError`，源已耗尽抛 ``RuntimeError``。
    """
    source = create_source(config)
    with source:
        frame = source.read()
    if frame is None:
        raise RuntimeError(f"capture returned no frame from {source.name}")
    return frame


# ===== 内置源注册 =====
def _build_camera(cfg: CaptureConfig, pre: Preprocessor | None) -> FrameSource:
    return CameraSource(cfg.camera, pre)


def _build_video(cfg: CaptureConfig, pre: Preprocessor | None) -> FrameSource:
    return VideoFileSource(cfg.video, pre)


def _build_image(cfg: CaptureConfig, pre: Preprocessor | None) -> FrameSource:
    return ImageSource(cfg.image.path, cfg.image.loop, pre)


def _build_synthetic(cfg: CaptureConfig, pre: Preprocessor | None) -> FrameSource:
    s = cfg.synthetic
    return SyntheticSource(
        width=s.width,
        height=s.height,
        fps=s.fps,
        total=s.total,
        pattern=s.pattern,
        preprocessor=pre,
    )


register_source("camera", _build_camera)
register_source("video", _build_video)
register_source("image", _build_image)
register_source("synthetic", _build_synthetic)
