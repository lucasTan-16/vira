"""ppe_rules 单元测试：用合成框构造场景，验证空间匹配规则。
Person 框 [100,100,200,300]：宽 100，高 200，中心 (150,200)。
归一化高度 ny = (装备中心y - 100) / 200。
"""
import numpy as np
from ppe_rules import evaluate

P = [100, 100, 200, 300]          # Person 框

# 各装备框（中心在 Person 对应区域）
helmet  = [135, 110, 165, 130]    # 中心(150,120) ny=0.10  头顶
goggles = [140, 125, 160, 135]    # 中心(150,130) ny=0.15  脸
vest    = [105, 150, 195, 220]    # 中心(150,185) ny=0.425 躯干，IoU 0.315
gloves  = [113, 213, 128, 228]    # 中心(120,220) ny=0.60  手
boots   = [128, 270, 153, 290]    # 中心(140,280) ny=0.90  脚


def _det(box, cls, conf=0.9):
    return [box[0], box[1], box[2], box[3], conf, cls]


# 1. 全装备佩戴 → 无违规
def test_all_worn():
    dets = [_det(P, 0), _det(helmet, 1), _det(goggles, 5),
            _det(vest, 3), _det(gloves, 2), _det(boots, 4)]
    results, violations = evaluate(np.array(dets))
    assert len(results) == 1
    assert all(results[0]["wear"].values())
    assert violations == []


# 2. 缺 helmet → 报缺失
def test_missing_helmet():
    dets = [_det(P, 0), _det(goggles, 5), _det(vest, 3),
            _det(gloves, 2), _det(boots, 4)]
    results, violations = evaluate(np.array(dets))
    assert results[0]["wear"]["helmet"] is False
    assert (0, "helmet") in violations


# 3. helmet 框在 Person 底部（位置错）→ 不算戴
def test_helmet_wrong_position():
    bad = [135, 270, 165, 290]      # 中心(150,280) ny=0.90
    dets = [_det(P, 0), _det(bad, 1)]
    results, _ = evaluate(np.array(dets))
    assert results[0]["wear"]["helmet"] is False


# 4. 一个 helmet 只归属一个人（两人一帽）
def test_one_ppe_one_person():
    p2 = [300, 100, 400, 300]
    dets = [_det(P, 0), _det(p2, 0), _det(helmet, 1)]
    results, _ = evaluate(np.array(dets))
    worn = [r["wear"]["helmet"] for r in results]
    assert worn == [True, False]


# 5. vest 位置对但面积太小（IoU 不足）→ 不算穿
def test_vest_needs_iou():
    small = [148, 168, 152, 172]    # 中心(150,170) ny=0.35 但 4x4
    dets = [_det(P, 0), _det(small, 3)]
    results, _ = evaluate(np.array(dets))
    assert results[0]["wear"]["vest"] is False
