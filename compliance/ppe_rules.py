"""PPE 合规规则层（确定性几何规则，非模型、非 LLM）。
输入视觉检测框（Person + 5 类 PPE，xyxy/conf/cls），
通过人-装备空间匹配，输出每个 Person 的佩戴状态与违规事件。
可单测、可解释、可审计。
"""
import numpy as np

# ========== 0. 类别与规则常量 ==========
# 视觉类别顺序：Person=0, helmet=1, gloves=2, vest=3, boots=4, goggles=5
CLS_TO_NAME = {1: "helmet", 2: "gloves", 3: "vest", 4: "boots", 5: "goggles"}
# 规则层输出的固定装备顺序
PPE_NAMES = ("helmet", "vest", "gloves", "boots", "goggles")

# 每类装备的匹配区域：PPE 框中心相对 Person 框的归一化高度范围
# （0=Person 顶，1=Person 底；适当放宽到框外，因帽子/脚可能超出）
RULES = {
    "helmet":  (-0.10, 0.20),   # 头顶，可能略超出框上沿
    "goggles": (0.00, 0.25),    # 脸部，比 helmet 略低
    "vest":    (0.15, 0.60),   # 躯干，另需 IoU
    "gloves":  (0.30, 0.90),   # 手，身体两侧下部
    "boots":   (0.75, 1.05),   # 脚，底部，可能略超出
}
VEST_IOU = 0.30                 # vest 与 Person 的最小 IoU


# ========== 1. 基础几何：框中心 / 归一化位置 / IoU ==========
def _center(box):
    """xyxy 框中心 (cx, cy)。"""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _norm_pos(person, ppe):
    """PPE 框中心相对 Person 框的归一化坐标 (nx, ny)。
    0~1 表示在 Person 框内，略超出为负数或 >1。
    """
    ecx, ecy = _center(ppe)
    pw = person[2] - person[0]
    ph = person[3] - person[1]
    nx = (ecx - person[0]) / pw
    ny = (ecy - person[1]) / ph
    return nx, ny


def _iou(a, b):
    """两个 xyxy 框的 IoU（交并比）。"""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


# ========== 2. 主函数：检测框 → 佩戴状态 + 违规 ==========
def evaluate(detections):
    """
    detections: numpy (M,6)，每行 [x1,y1,x2,y2,conf,cls]
    返回：
      results: list[dict]，每人 {person, wear{name:bool}, missing[...]}
      violations: list[(person_id, 装备名)]，扁平化的违规清单
    """
    # 分离 Person 与 PPE；PPE 带 used 标志，保证一个装备只归属一人
    persons, ppe_items = [], []
    for det in detections:
        box, cls = det[:4], int(det[5])
        if cls == 0:
            persons.append((len(persons), box))
        else:
            ppe_items.append([box, cls, False])

    results = []
    for pid, pbox in persons:
        wear = {name: False for name in PPE_NAMES}
        for k, (ebox, ecls, used) in enumerate(ppe_items):
            if used:
                continue
            name = CLS_TO_NAME.get(ecls)
            if name is None or wear[name]:
                continue
            nx, ny = _norm_pos(pbox, ebox)
            cy_lo, cy_hi = RULES[name]
            horiz = -0.15 <= nx <= 1.15              # 水平在框内（略放宽）
            vert = cy_lo <= ny <= cy_hi              # 高度落在对应区域
            iou_ok = _iou(pbox, ebox) >= VEST_IOU if name == "vest" else True
            if horiz and vert and iou_ok:
                wear[name] = True
                ppe_items[k][2] = True              # 标记已归属

        missing = [n for n in PPE_NAMES if not wear[n]]
        results.append({"person": pid, "wear": wear, "missing": missing})

    # 违规：把每个人缺失的装备展开成 (人, 装备) 列表
    violations = [(r["person"], m) for r in results for m in r["missing"]]
    return results, violations
