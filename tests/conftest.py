"""pytest 共享夹具：把仓库根加入 sys.path，并提供无硬件依赖的合成源工厂。"""
from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture()
def synthetic_config():
    """小分辨率合成源配置，跑得快、不占内存。"""
    from capturekit import CaptureConfig

    return CaptureConfig.from_dict(
        {"kind": "synthetic", "synthetic": {"width": 64, "height": 48, "fps": 0.0}}
    )
