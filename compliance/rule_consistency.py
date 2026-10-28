"""B2 规则层一致率验证：规则层判定 vs 原始 no_xxx 标注。
主用 FP32（best.pt）检测，排除检测误差，纯验证规则层。
输出：样本数、总体一致率、5 类各自一致率、Kappa、错判案例。
"""
from pathlib import Path
import numpy as np
import sys
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppe_rules import evaluate

# ========== 0. 路径与类别配置 ==========
ROOT = Path.home() / "vira"
RAW_DIR = ROOT / "vision/datasets"                 # 原始 11 类数据集
BEST = ROOT / "vision/runs/ppe6_yolo11n/weights/best.pt"
TEST_IMG = RAW_DIR / "images/test"
TEST_LBL = RAW_DIR / "labels/test"

# 原始 11 类索引
RAW_HELMET, RAW_GLOVES, RAW_VEST, RAW_BOOTS, RAW_GOGGLES = 0, 1, 2, 3, 4
RAW_NONE, RAW_PERSON = 5, 6
RAW_NO_HELMET, RAW_NO_GOGGLE, RAW_NO_GLOVES, RAW_NO_BOOTS = 7, 8, 9, 10

# 装备名 → (正类=戴, 负类=没戴)；vest 没戴用 none
ITEM_MAP = {
    "helmet":  (RAW_HELMET,  RAW_NO_HELMET),
    "goggles": (RAW_GOGGLES, RAW_NO_GOGGLE),
    "vest":    (RAW_VEST,    RAW_NONE),
    "gloves":  (RAW_GLOVES,  RAW_NO_GLOVES),
    "boots":   (RAW_BOOTS,   RAW_NO_BOOTS),
}
PPE_ORDER = ("helmet", "vest", "gloves", "boots", "goggles")


# ========== 1. GT 标签解析：YOLO 归一化 → 像素框 ==========
def load_gt(label_path, img_w, img_h):
    """读原始标签，返回 (N,5)：x1,y1,x2,y2,cls。"""
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, w, h = map(float, p[:5])
            x1 = (cx - w / 2) * img_w; y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w; y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2, int(c)])
    return np.array(boxes) if boxes else np.zeros((0, 5))
# ========== 2. FP32 检测 + 规则层 ==========
def run_detect(model, img_path):
    """FP32 检测，返回 (M,6)：x1,y1,x2,y2,conf,cls（6 类）。"""
    r = model.predict(img_path, verbose=False, conf=0.25, iou=0.7)[0]
    return r.boxes.data.cpu().numpy()


def gt_person_states(gt):
    """对每个 GT Person，按位置匹配正/负类框，得到逐装备 GT 状态。
    返回 list[dict]：{box, state{name: True/False/None}}（None=未标注/不可见）。
    """
    persons = gt[gt[:, 4] == RAW_PERSON]
    out = []
    for p in persons:
        state = {n: None for n in PPE_ORDER}
        for item in PPE_ORDER:
            pos_cls, neg_cls = ITEM_MAP[item]
            for b in gt:
                if int(b[4]) not in (pos_cls, neg_cls):
                    continue
                cx = (b[0] + b[2]) / 2; cy = (b[1] + b[3]) / 2
                # 装备框中心落在 Person 框内才算这个人的
                if p[0] <= cx <= p[2] and p[1] <= cy <= p[3]:
                    state[item] = (int(b[4]) == pos_cls)
                    break
        out.append({"box": p[:4], "state": state})
    return out
# ========== 3. Person 匹配 + 逐类对比 ==========
def _iou(a, b):
    """两个 xyxy 框的 IoU。"""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def process_image(model, img_path, lbl_path):
    """处理单图，返回 (图名, 装备, GT状态, 规则层状态) 列表。
    没检测到这个人时规则层状态为 None（漏检）。
    """
    from PIL import Image
    w, h = Image.open(img_path).size
    gt = load_gt(lbl_path, w, h)
    gt_persons = gt_person_states(gt)

    dets = run_detect(model, img_path)
    pred_results, _ = evaluate(dets)
    det_persons = dets[dets[:, 5] == 0]   # 与 pred_results 同序

    # 一对一贪心匹配：算所有 (GT,检测) IoU，按 IoU 降序占用
    cand = []
    for gi, gp in enumerate(gt_persons):
        for di, dp in enumerate(det_persons):
            v = _iou(gp["box"], dp[:4])
            if v >= 0.5:
                cand.append((v, gi, di))
    cand.sort(reverse=True)
    g_used, d_used = {}, {}
    for v, gi, di in cand:
        if gi not in g_used and di not in d_used:
            g_used[gi], d_used[di] = di, gi

    pairs = []
    for gi, gp in enumerate(gt_persons):
        best_i = g_used.get(gi)
        if best_i is None:
            # 漏检：GT 有标注的装备，规则层无法判，记 None
            for item in PPE_ORDER:
                if gp["state"][item] is not None:
                    pairs.append((img_path.name, item, gp["state"][item], None))
            continue
        wear = pred_results[best_i]["wear"]
        for item in PPE_ORDER:
            if gp["state"][item] is not None:
                pairs.append((img_path.name, item, gp["state"][item], wear[item]))
    return pairs

# ========== 4. 统计 + 主函数 ==========
def summarize(all_pairs):
    """返回总体/逐类一致率、Kappa、错判、漏检。"""
    scored = [p for p in all_pairs if p[3] is not None]
    missed = len(all_pairs) - len(scored)
    n = len(scored)

    agree = sum(1 for p in scored if p[2] == p[3])
    overall = agree / n if n else 0

    per_item = {}
    for item in PPE_ORDER:
        ps = [p for p in scored if p[1] == item]
        a = sum(1 for p in ps if p[2] == p[3])
        per_item[item] = (a / len(ps) if ps else 0, len(ps))

    # 2x2 混淆矩阵（True=戴，False=没戴）
    tp = sum(1 for p in scored if p[2] and p[3])
    tn = sum(1 for p in scored if (not p[2]) and (not p[3]))
    fp = sum(1 for p in scored if (not p[2]) and p[3])
    fn = sum(1 for p in scored if p[2] and (not p[3]))
    po = (tp + tn) / n if n else 0
    p_yes = (tp + fn) / n if n else 0
    p_py = (tp + fp) / n if n else 0
    pe = p_yes * p_py + (1 - p_yes) * (1 - p_py)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0

    errors = [p for p in scored if p[2] != p[3]]
    return {"n": n, "missed": missed, "overall": overall, "per_item": per_item,
            "kappa": kappa, "cm": (tp, tn, fp, fn), "errors": errors}


def main():
    model = YOLO(BEST)
    all_pairs = []
    for img_path in sorted(TEST_IMG.glob("*.jpg")):
        lbl = TEST_LBL / (img_path.stem + ".txt")
        all_pairs.extend(process_image(model, img_path, lbl))

    s = summarize(all_pairs)
    print("=== B2 规则层一致率（FP32 检测，纯验证规则层）===")
    print(f"样本判定数: {s['n']}（漏检 {s['missed']}）")
    print(f"总体一致率: {s['overall']*100:.2f}%")
    print(f"Cohen Kappa: {s['kappa']:.3f}")
    tp, tn, fp, fn = s["cm"]
    print(f"混淆矩阵 TP={tp} TN={tn} FP={fp} FN={fn}")
    print("逐类一致率:")
    for item, (rate, k) in s["per_item"].items():
        print(f"  {item:8s}: {rate*100:6.2f}%  (n={k})")
    print(f"错判案例: {len(s['errors'])} 个（前 20）")
    for name, item, g, p in s["errors"][:20]:
        print(f"  {name} {item:8s} GT={'戴' if g else '没戴'} 规则层={'戴' if p else '没戴'}")


if __name__ == "__main__":
    main()
