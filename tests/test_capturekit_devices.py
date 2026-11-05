"""设备探测与配置示例测试：无摄像头环境下也必须给出可读结果。"""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from capturekit import (  # noqa: E402
    CaptureConfig,
    SourceConfigError,
    absence_hint,
    has_camera,
    probe_camera_devices,
)


def test_probe_returns_list_and_never_raises():
    devices = probe_camera_devices()
    assert isinstance(devices, list)
    for item in devices:
        assert item.startswith("/dev/video")


def test_has_camera_matches_probe():
    assert has_camera() == bool(probe_camera_devices())


def test_absence_hint_covers_actionable_steps():
    hint = absence_hint()
    for token in ("lsusb", "dmesg", "synthetic"):
        assert token in hint


def test_camera_config_rejects_bad_values():
    with pytest.raises(SourceConfigError):
        CaptureConfig.from_dict({"kind": "camera", "camera": {"width": 0}}).validate()
    with pytest.raises(SourceConfigError):
        CaptureConfig.from_dict({"kind": "camera", "camera": {"fourcc": "MJ"}}).validate()


def test_config_roundtrip_through_json(tmp_path):
    cfg = CaptureConfig.from_dict(
        {"kind": "camera", "camera": {"index": 0, "width": 640, "height": 480, "fps": 30}}
    )
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(cfg.to_dict(), ensure_ascii=False), encoding="utf-8")
    loaded = CaptureConfig.from_file(str(path))
    assert loaded.kind == "camera"
    assert loaded.camera.width == 640 and loaded.camera.fps == 30
