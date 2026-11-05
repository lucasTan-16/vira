"""采集层配置：带校验的 dataclass 配置树，支持从 dict/JSON/YAML 加载。

原则：
- **默认值可跑**——``CaptureConfig()`` 就是板端默认摄像头配置；
- **非法配置启动期报错**——越界参数（如分辨率 <=0、buffer_size <=0）在
  :meth:`CaptureConfig.validate` 抛出 :class:`SourceConfigError`，不带病运行；
- **未知字段报错**——避免拼写错误被静默忽略导致跑的是默认参数。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from .errors import SourceConfigError
from .preprocess import PreprocessConfig

SUPPORTED_KINDS = ("camera", "video", "image", "synthetic")


def _strict_kwargs(cls, data: dict[str, Any], where: str) -> dict[str, Any]:
    """把 dict 转成 dataclass 关键字参数，未知键直接报错。"""
    if not isinstance(data, dict):
        raise SourceConfigError(f"{where}: expect mapping, got {type(data).__name__}")
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise SourceConfigError(f"{where}: unknown field(s) {sorted(unknown)}; allowed {sorted(allowed)}")
    return dict(data)


@dataclass(frozen=True)
class CameraConfig:
    """摄像头参数（UVC / V4L2）。"""

    index: int = 0
    width: int = 640
    height: int = 480
    fourcc: str = "MJPG"
    fps: int = 30
    buffer_size: int = 1
    backend: str = "v4l2"
    max_read_failures: int = 5

    def validate(self) -> None:
        if self.index < 0:
            raise SourceConfigError(f"camera.index must be >= 0, got {self.index}")
        if self.width <= 0 or self.height <= 0:
            raise SourceConfigError(
                f"camera resolution must be positive, got {self.width}x{self.height}"
            )
        if len(self.fourcc) != 4:
            raise SourceConfigError(f"camera.fourcc must be 4 chars, got {self.fourcc!r}")
        if self.fps <= 0:
            raise SourceConfigError(f"camera.fps must be > 0, got {self.fps}")
        if self.buffer_size < 1:
            raise SourceConfigError(f"camera.buffer_size must be >= 1, got {self.buffer_size}")
        if self.max_read_failures < 1:
            raise SourceConfigError(
                f"camera.max_read_failures must be >= 1, got {self.max_read_failures}"
            )


@dataclass(frozen=True)
class VideoConfig:
    """视频文件参数。"""

    path: str = ""
    loop: bool = False
    realtime: bool = True
    max_read_failures: int = 5

    def validate(self) -> None:
        if not self.path:
            raise SourceConfigError("video.path must be non-empty")
        if self.max_read_failures < 1:
            raise SourceConfigError("video.max_read_failures must be >= 1")


@dataclass(frozen=True)
class ImageConfig:
    """静态图像参数。"""

    path: str = ""
    loop: bool = True

    def validate(self) -> None:
        if not self.path:
            raise SourceConfigError("image.path must be non-empty")


@dataclass(frozen=True)
class SyntheticConfig:
    """合成源参数。"""

    width: int = 640
    height: int = 480
    fps: float = 30.0
    total: int = 0
    pattern: str = "gradient"

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise SourceConfigError("synthetic resolution must be positive")
        if self.total < 0:
            raise SourceConfigError("synthetic.total must be >= 0 (0 = infinite)")
        if self.pattern not in ("gradient", "noise", "fill"):
            raise SourceConfigError(f"synthetic.pattern unsupported: {self.pattern}")


@dataclass
class CaptureConfig:
    """采集配置根节点。

    ``kind`` 决定哪个子配置生效，其余子配置保留默认值不影响运行。

    Example:
        >>> CaptureConfig.from_dict({"kind": "camera", "camera": {"index": 0}})
        CaptureConfig(kind='camera', ...)
    """

    kind: str = "camera"
    camera: CameraConfig = field(default_factory=CameraConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

    # ===== 构造 =====
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaptureConfig":
        """从 dict 构造并校验。"""
        kwargs = _strict_kwargs(cls, data or {}, "capture")
        pre_kwargs = _strict_kwargs(PreprocessConfig, kwargs.get("preprocess", {}), "capture.preprocess")
        if "target_size" in pre_kwargs:
            pre_kwargs["target_size"] = tuple(int(v) for v in pre_kwargs["target_size"])  # type: ignore[assignment]
        cfg = cls(
            kind=kwargs.get("kind", "camera"),
            camera=CameraConfig(**_strict_kwargs(CameraConfig, kwargs.get("camera", {}), "capture.camera")),
            video=VideoConfig(**_strict_kwargs(VideoConfig, kwargs.get("video", {}), "capture.video")),
            image=ImageConfig(**_strict_kwargs(ImageConfig, kwargs.get("image", {}), "capture.image")),
            synthetic=SyntheticConfig(
                **_strict_kwargs(SyntheticConfig, kwargs.get("synthetic", {}), "capture.synthetic")
            ),
            preprocess=PreprocessConfig(**pre_kwargs),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_file(cls, path: str) -> "CaptureConfig":
        """从 JSON 文件加载；存在 PyYAML 时也支持 .yaml/.yml。"""
        if not os.path.exists(path):
            raise SourceConfigError(f"config file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            if path.lower().endswith((".yaml", ".yml")):
                try:
                    import yaml  # 可选依赖，板端未安装时退回 JSON
                except ImportError as exc:  # pragma: no cover - 环境相关
                    raise SourceConfigError(
                        "PyYAML not installed; use JSON config instead"
                    ) from exc
                data = yaml.safe_load(fh) or {}
            else:
                try:
                    data = json.load(fh)
                except json.JSONDecodeError as exc:
                    raise SourceConfigError(f"invalid JSON config {path}: {exc}") from exc
        return cls.from_dict(data)

    # ===== 校验 =====
    def active(self) -> CameraConfig | VideoConfig | ImageConfig | SyntheticConfig:
        """返回当前 ``kind`` 对应的子配置。"""
        return getattr(self, self.kind)

    def validate(self) -> None:
        if self.kind not in SUPPORTED_KINDS:
            raise SourceConfigError(
                f"capture.kind must be one of {SUPPORTED_KINDS}, got {self.kind!r}"
            )
        self.active().validate()
        self.preprocess.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        """一行摘要，用于日志与结果表头。"""
        active = self.active()
        body = ", ".join(f"{k}={v}" for k, v in asdict(active).items() if v not in ("", None))
        return f"{self.kind}({body})"
