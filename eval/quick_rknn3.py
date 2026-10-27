import sys
import cv2
import numpy as np
from rknn.api import RKNN

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)
CLASSES = ("Person", "helmet", "gloves", "vest", "boots", "goggles")

ROOT = "/home/lucas/vira"
ONNX = ROOT + "/vision/runs/ppe6_yolo11n/weights/best_rknnopt.onnx"
DATASET = ROOT + "/vision/rknn/calib/dataset.txt"


def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[pos]
    return boxes[pos], classes[pos], scores


def nms_boxes(boxes, scores):
    x = boxes[:, 0]; y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]; h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 1e-5)
        h1 = np.maximum(0.0, yy2 - yy1 + 1e-5)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)


def dfl(position):
    import torch
    x = torch.tensor(position)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w).softmax(2)
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
    box_xy = grid + 0.5 - position[:, 0:2]
    box_xy2 = grid + 0.5 + position[:, 2:4]
    return np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)


def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    branches = 3
    ppb = len(input_data) // branches
    for i in range(branches):
        boxes.append(box_process(input_data[ppb * i]))
        classes_conf.append(input_data[ppb * i + 1])
        scores.append(np.ones_like(input_data[ppb * i + 1][:, :1], dtype=np.float32))

    def flat(_in):
        ch = _in.shape[1]
        return _in.transpose(0, 2, 3, 1).reshape(-1, ch)

    boxes = np.concatenate([flat(v) for v in boxes])
    classes_conf = np.concatenate([flat(v) for v in classes_conf])
    scores = np.concatenate([flat(v) for v in scores])
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b, cl, s = boxes[inds], classes[inds], scores[inds]
        keep = nms_boxes(b, s)
        if len(keep):
            nboxes.append(b[keep]); nclasses.append(cl[keep]); nscores.append(s[keep])
    if not nclasses:
        return None, None, None
    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)


def letterbox(im, new_shape=(640, 640), pad=(0, 0, 0)):
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    imr = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape[0] - nh) // 2
    left = (new_shape[1] - nw) // 2
    return cv2.copyMakeBorder(imr, top, new_shape[0] - nh - top, left,
                              new_shape[1] - nw - left, cv2.BORDER_CONSTANT, value=pad)


def main():
    mode, img_path = sys.argv[1], sys.argv[2]
    rknn = RKNN()
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588")
    rknn.load_onnx(model=ONNX)
    if mode == "fp":
        rknn.build(do_quantization=False)
    else:
        rknn.build(do_quantization=True, dataset=DATASET)
    rknn.init_runtime()

    img = cv2.imread(img_path)
    lb = letterbox(img, (640, 640), (0, 0, 0))
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    inp = rgb[np.newaxis, ...]  # NHWC 1,640,640,3, 0-255

    outputs = rknn.inference(inputs=[inp])
    print("mode:", mode, "num outputs:", len(outputs))
    for i, o in enumerate(outputs):
        print(i, o.shape, o.dtype, "min/max:", round(float(o.min()), 4), round(float(o.max()), 4))

    boxes, classes, scores = post_process(outputs)
    print("=== result ===")
    if boxes is None:
        print("no boxes (cls 全崩 或 需 sigmoid)")
    else:
        for b, c, s in sorted(zip(boxes, classes, scores), key=lambda x: -x[2])[:20]:
            print(f"{CLASSES[c]:8s} {float(s):.3f}  {b.astype(int).tolist()}")
    rknn.release()


if __name__ == "__main__":
    main()
