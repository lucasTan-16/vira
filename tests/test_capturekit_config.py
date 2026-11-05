"""配置层测试：默认值可用、非法配置启动期报错、未知字段不被静默忽略。"""
from __future__ import annotations

import json

import pytest

from capturekit import CameraConfig, CaptureConfig, SourceConfigError


def test_default_config_is_camera():
    cfg = CaptureConfig()
    cfg.validate()
    assert cfg.kind == "camera"
    assert isinstance(cfg.active(), CameraConfig)
    assert cfg.camera.index == 0


def test_camera_defaults_match_board_contract():
    """板端既有口径：MJPG 640x480、驱动缓冲 1。"""
    cam = CameraConfig()
    assert (cam.width, cam.height) == (640, 480)
    assert cam.fourcc == "MJPG"
    assert cam.buffer_size == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "camera", "camera": {"width": 0}},
        {"kind": "camera", "camera": {"height": -1}},
        {"kind": "camera", "camera": {"fps": 0}},
        {"kind": "camera", "camera": {"buffer_size": 0}},
        {"kind": "camera", "camera": {"fourcc": "MJ"}},
        {"kind": "camera", "camera": {"index": -1}},
        {"kind": "nope"},
    ],
)
def test_invalid_config_raises(bad):
    with pytest.raises(SourceConfigError):
        CaptureConfig.from_dict(bad)


def test_unknown_field_not_silently_ignored():
    with pytest.raises(SourceConfigError) as exc:
        CaptureConfig.from_dict({"kind": "camera", "camera": {"resolutoin": 640}})
    assert "unknown field" in str(exc.value)


def test_from_file_json_roundtrip(tmp_path):
    payload = {
        "kind": "camera",
        "camera": {"index": 2, "width": 1280, "height": 720, "fps": 15},
    }
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = CaptureConfig.from_file(str(path))
    assert cfg.kind == "camera"
    assert cfg.camera.index == 2
    assert (cfg.camera.width, cfg.camera.height) == (1280, 720)
    assert cfg.to_dict()["camera"]["fps"] == 15


def test_from_file_missing_raises(tmp_path):
    with pytest.raises(SourceConfigError):
        CaptureConfig.from_file(str(tmp_path / "absent.json"))


def test_summary_is_single_line():
    text = CaptureConfig().summary()
    assert "\n" not in text and "camera" in text
