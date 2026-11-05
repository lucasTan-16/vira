"""采集源测试：抽象契约、预处理、故障语义。

全部基于合成源 / 临时视频文件 / 临时图片，**不依赖摄像头硬件**（C8 场景下也能全绿）。
"""
from __future__ import annotations

import numpy as np
import pytest

from capturekit import (
    CaptureConfig,
    DeviceUnavailableError,
    Frame,
    FrameReadError,
    ImageSource,
    INPUT_SIZE,
    SourceClosedError,
    SyntheticSource,
    VideoFileSource,
    VideoConfig,
    create_source,
    grab_frame,
    supported_kinds,
)


# ===== 抽象契约 =====
def test_synthetic_frame_contract():
    cfg = CaptureConfig.from_dict(
        {"kind": "synthetic", "synthetic": {"width": 320, "height": 240, "total": 3}}
    )
    with create_source(cfg) as source:
        frames = list(source)
    assert len(frames) == 3
    first = frames[0]
    assert isinstance(first, Frame)
    assert first.seq == 0                       # 序号从 0 连续自增
    assert first.image.shape == (*INPUT_SIZE, 3)  # letterbox 到模型输入尺寸
    assert first.image.dtype == np.uint8
    assert first.source == "synthetic"
    assert first.meta["raw_shape"] == (240, 320, 3)
    assert [f.seq for f in frames] == [0, 1, 2]


def test_stale_seq_exposes_dropped_frames():
    """seq 连续递增，消费侧可通过跳号识别"中间帧被丢掉了多少"。"""
    source = SyntheticSource(width=32, height=32, fps=0, total=5)
    with source:
        seqs = [source.read().seq for _ in range(5)]
    assert seqs == [0, 1, 2, 3, 4]


def test_read_before_open_raises():
    source = SyntheticSource()
    with pytest.raises(SourceClosedError):
        source.read()


def test_close_is_idempotent_and_reopen_resets_seq():
    source = SyntheticSource(width=16, height=16, fps=0, total=1)
    source.open()
    assert source.read().seq == 0
    source.close()
    source.close()                              # 幂等，不抛
    assert source.is_open is False

    source.open()                               # 重开：会话内统计与序号复位
    assert source.read().seq == 0
    source.close()


def test_context_manager_closes_on_exception():
    source = SyntheticSource(width=16, height=16, fps=0, total=2)
    with pytest.raises(RuntimeError):
        with source:
            raise RuntimeError("boom")
    assert source.is_open is False


def test_stats_track_reads_and_uptime():
    with SyntheticSource(width=16, height=16, fps=0, total=3) as source:
        for _ in range(3):
            source.read()
        stats = source.stats.as_dict()
    assert stats["frames_read"] == 3
    assert stats["read_errors"] == 0
    assert stats["uptime_s"] >= 0.0


# ===== 故障语义 =====
def test_injected_read_failures_escalate_to_device_unavailable():
    """连续读失败达到上限 → 升级为不可自愈的设备故障，避免上层无意义重试。"""
    source = SyntheticSource(width=16, height=16, fps=0, fail_every=1)
    source.max_read_failures = 2
    with source:
        assert source.read() is not None        # seq=0 正常产帧（fail_every 从 seq>0 起生效）
        with pytest.raises(FrameReadError):
            source.read()                       # 第 1 次读失败
        with pytest.raises(DeviceUnavailableError):
            source.read()                       # 第 2 次失败 → 达上限升级
    assert source.stats.read_errors >= 2


def test_infinite_synthetic_source_never_ends():
    source = SyntheticSource(width=16, height=16, fps=0, total=0)
    with source:
        for _ in range(50):
            assert source.read() is not None


# ===== 摄像头不可用时的行为（C8）=====
def test_camera_source_fails_fast_with_actionable_message():
    """设备不存在时必须快速失败并给出排查方向，而不是返回空帧。"""
    from capturekit import CameraConfig, CameraSource

    cam = CameraSource(CameraConfig(index=99))
    with pytest.raises(DeviceUnavailableError) as exc:
        cam.open()
    assert "cannot open camera 99" in str(exc.value)


def test_camera_probe_reports_unavailable():
    from capturekit import CameraConfig, CameraSource

    info = CameraSource.probe(CameraConfig(index=99))
    assert info["available"] is False
    assert info["device"].endswith("video99")


# ===== 视频 / 图像源 =====
def _write_video(path, frames=5, size=(64, 48)):
    import cv2

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, size
    )
    assert writer.isOpened(), "cv2.VideoWriter unavailable in this build"
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), i * 20 % 256, dtype=np.uint8))
    writer.release()


def test_video_source_reads_all_frames_then_eof(tmp_path):
    path = tmp_path / "clip.avi"
    _write_video(path, frames=5)

    source = VideoFileSource(VideoConfig(path=str(path), loop=False, realtime=False))
    with source:
        count = 0
        while source.read() is not None:
            count += 1
    assert count == 5


def test_video_source_loops_when_configured(tmp_path):
    path = tmp_path / "clip.avi"
    _write_video(path, frames=3)

    source = VideoFileSource(VideoConfig(path=str(path), loop=True, realtime=False))
    with source:
        seqs = [source.read().seq for _ in range(7)]
    assert seqs == list(range(7))               # 循环后仍在产帧，不中断


def test_video_source_missing_file_raises(tmp_path):
    source = VideoFileSource(VideoConfig(path=str(tmp_path / "absent.mp4")))
    with pytest.raises(DeviceUnavailableError):
        source.open()


def test_image_source_repeats_and_can_be_exhausted(tmp_path):
    import cv2

    path = tmp_path / "still.png"
    cv2.imwrite(str(path), np.zeros((48, 64, 3), dtype=np.uint8))

    looping = ImageSource(str(path), loop=True)
    with looping:
        assert looping.read() is not None
        assert looping.read() is not None

    single = ImageSource(str(path), loop=False)
    with single:
        assert single.read() is not None
        assert single.read() is None            # 非循环：第二帧即 EOF


# ===== 工厂 =====
def test_registry_exposes_all_kinds():
    assert set(supported_kinds()) == {"camera", "image", "synthetic", "video"}


def test_create_source_returns_expected_type():
    assert isinstance(create_source(CaptureConfig.from_dict({"kind": "synthetic"})), SyntheticSource)


def test_grab_frame_returns_single_processed_frame():
    frame = grab_frame(CaptureConfig.from_dict({"kind": "synthetic"}))
    assert frame.image.shape == (*INPUT_SIZE, 3)
    assert frame.seq == 0
