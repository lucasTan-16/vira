import sys, os, glob
import cv2
import numpy as np
import torch
import onnxruntime
from ultralytics.utils.metrics import ap_per_class
from ultralytics.utils.ops import non_max_suppression, scale_boxes

OBJ_THRESH = 0.001
IOUV = np.linspace(0.5, 0.95, 10)
CLASSES = ("Person", "helmet", "gloves", "vest", "boots", "goggles")

ROOT = "/home/lucas/vira"
ONNX = ROOT + "/vision/runs/ppe6_yolo11n/weights/best_full.onnx"
TEST_IMG = ROOT + "/vision/datasets_ppe6/images/test"
TEST_LBL = ROOT + "/vision/datasets_ppe6/labels/test"


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw = (new_shape[1] - nw) / 2
    dh = (new_shape[0] - nh) / 2
    if (w, h) != (nw, nh):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (dw, dh)


def box_iou(b1, b2):
    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-9)


def match_predictions(pred_cls, true_cls, iou):
    N = pred_cls.shape[0]
    correct = np.zeros((N, len(IOUV)), dtype=bool)
    if N == 0 or true_cls.shape[0] == 0:
        return correct
    same = pred_cls[:, None] == true_cls[None, :]
    iou = iou * same
    for j, th in enumerate(IOUV):
        m = np.argwhere(iou >= th)
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
            correct[m[:, 0].astype(int), j] = True
    return correct


def read_gt(path, iw, ih):
    if not os.path.exists(path):
        return np.array([]), np.zeros((0, 4))
    cls, box = [], []
    for line in open(path):
        p = line.split()
        c = int(p[0]); xc, yc, w, h = map(float, p[1:])
        box.append([(xc - w / 2) * iw, (yc - h / 2) * ih,
                    (xc + w / 2) * iw, (yc + h / 2) * ih])
        cls.append(c)
    return np.array(cls), np.array(box, dtype=np.float32).reshape(-1, 4)


def main():
    sess = onnxruntime.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    imgs = sorted(glob.glob(TEST_IMG + "/*.jpg")) + sorted(glob.glob(TEST_IMG + "/*.jpeg"))
    print("num test images:", len(imgs))
    T, C, PC, TC = [], [], [], []
    for k, ip in enumerate(imgs):
        img = cv2.imread(ip)
        ih, iw = img.shape[:2]
        lb, ratio, pad = letterbox(img, (640, 640))
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        x = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = sess.run(None, {"images": x})
        det = non_max_suppression(torch.from_numpy(out[0]), conf_thres=OBJ_THRESH,
                                  iou_thres=0.7, multi_label=True, max_det=300)[0]

        name = os.path.splitext(os.path.basename(ip))[0]
        tcls, tbox = read_gt(TEST_LBL + "/" + name + ".txt", iw, ih)
        if len(det):
            det = det.clone()
            scale_boxes(lb.shape, det[:, :4], img.shape, ratio_pad=(ratio, pad))
            boxes = det[:, :4].numpy()
            scores = det[:, 4].numpy()
            classes = det[:, 5].numpy().astype(int)
            iou = box_iou(boxes, tbox) if tbox.shape[0] else np.zeros((len(boxes), 0))
            correct = match_predictions(classes, tcls, iou)
            T.append(correct); C.append(scores); PC.append(classes)
        TC.append(tcls)
        print(f"{k+1}/{len(imgs)}", end="\r")

    tp = np.concatenate(T) if T else np.zeros((0, 10), bool)
    conf = np.concatenate(C) if C else np.array([])
    pcls = np.concatenate(PC) if PC else np.array([])
    tcls = np.concatenate(TC)

    res = ap_per_class(tp, conf, pcls, tcls, names=dict(enumerate(CLASSES)))
    p, r, ap, ucls = res[2], res[3], res[5], res[6]
    print("\n=== ONNX best_full (control, official NMS) ===")
    print(f"mAP50    {ap[:,0].mean():.4f}")
    print(f"mAP50-95 {ap.mean():.4f}")
    print(f"{'class':8s} {'mAP50':>7s} {'P':>7s} {'R':>7s}")
    for ci, c in enumerate(ucls):
        print(f"{CLASSES[c]:8s} {ap[ci,0]:7.4f} {p[ci]:7.4f} {r[ci]:7.4f}")


main()
