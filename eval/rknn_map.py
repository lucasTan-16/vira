import sys, os, glob
import cv2
import numpy as np
import torch
from rknn.api import RKNN
from ultralytics.utils.metrics import ap_per_class
from ultralytics.utils.ops import non_max_suppression, scale_boxes

ONNX = "/home/lucas/vira/vision/runs/ppe6_yolo11n/weights/best_rknnopt.onnx"
DATASET = "/home/lucas/vira/vision/rknn/calib/dataset.txt"
TEST_IMG = "/home/lucas/vira/vision/datasets_ppe6/images/test"
TEST_LBL = "/home/lucas/vira/vision/datasets_ppe6/labels/test"
IMG_SIZE = (640, 640)
IOUV = np.linspace(0.5, 0.95, 10)
CLASSES = ("Person", "helmet", "gloves", "vest", "boots", "goggles")


def dfl(position):
    x = torch.tensor(position)
    n, c, h, w = x.shape
    mc = c // 4
    y = x.reshape(n, 4, mc, h, w).softmax(2)
    acc = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(2).numpy()


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)
    position = dfl(position)
    xy = grid + 0.5 - position[:, 0:2]
    xy2 = grid + 0.5 + position[:, 2:4]
    return np.concatenate((xy * stride, xy2 * stride), axis=1)


def decode_three_head(input_data):
    boxes_out, cls_out = [], []
    ppb = len(input_data) // 3
    for i in range(3):
        boxes_out.append(box_process(input_data[ppb * i]))
        cls_out.append(input_data[ppb * i + 1])

    def flat(x):
        ch = x.shape[1]
        return x.transpose(0, 2, 3, 1).reshape(-1, ch)

    xyxy = np.concatenate([flat(x) for x in boxes_out])
    xywh = np.empty_like(xyxy)
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    cls = np.concatenate([flat(x) for x in cls_out])
    pred = np.concatenate([xywh, cls], axis=1).T[None]
    return pred.astype(np.float32)


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
    mode = sys.argv[1]
    rknn = RKNN()
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588")
    rknn.load_onnx(model=ONNX)
    if mode == "fp":
        rknn.build(do_quantization=False)
    else:
        rknn.build(do_quantization=True, dataset=DATASET)
    rknn.init_runtime()
    imgs = sorted(glob.glob(TEST_IMG + "/*.jpg")) + sorted(glob.glob(TEST_IMG + "/*.jpeg"))
    print("num test images:", len(imgs))
    T, C, PC, TC = [], [], [], []
    for k, ip in enumerate(imgs):
        img = cv2.imread(ip)
        ih, iw = img.shape[:2]
        lb, ratio, pad = letterbox(img)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        out = rknn.inference(inputs=[rgb[np.newaxis, ...]])
        pred = decode_three_head(out)
        det = non_max_suppression(torch.from_numpy(pred), conf_thres=0.001,
                                  iou_thres=0.7, multi_label=True, max_det=300)[0]
        name = os.path.splitext(os.path.basename(ip))[0]
        tcls, tbox = read_gt(TEST_LBL + "/" + name + ".txt", iw, ih)
        if len(det):
            det = det.clone()
            scale_boxes(lb.shape, det[:, :4], img.shape, ratio_pad=(ratio, pad))
            A = det.numpy()
            iou = box_iou(A[:, :4], tbox)
            correct = match_predictions(A[:, 5].astype(int), tcls, iou)
            T.append(correct); C.append(A[:, 4]); PC.append(A[:, 5].astype(int))
        TC.append(tcls)
        print(f"{k+1}/{len(imgs)}", end="\r")
    tp = np.concatenate(T) if T else np.zeros((0, 10), bool)
    conf = np.concatenate(C) if C else np.array([])
    pcls = np.concatenate(PC) if PC else np.array([])
    tcls = np.concatenate(TC)
    res = ap_per_class(tp, conf, pcls, tcls, names=dict(enumerate(CLASSES)))
    p, r, ap, ucls = res[2], res[3], res[5], res[6]
    print("\n=== mode:", mode, "===")
    print(f"mAP50    {ap[:,0].mean():.4f}")
    print(f"mAP50-95 {ap.mean():.4f}")
    print(f"{'class':8s} {'mAP50':>7s} {'P':>7s} {'R':>7s}")
    for ci, c in enumerate(ucls):
        print(f"{CLASSES[c]:8s} {ap[ci,0]:7.4f} {p[ci]:7.4f} {r[ci]:7.4f}")
    rknn.release()


main()
