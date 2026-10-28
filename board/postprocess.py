"""板端 YOLO 三头后处理（纯 numpy，无 torch/cv2）。
三头 RKNN 输出 9 个张量 → decode 成标准检测框 → numpy NMS。
供 e2e 计时、摄像头实时、板端服务复用。
"""
import numpy as np

# 输入图尺寸（与训练/转换一致）
IMG_SIZE = (640, 640)


# ========== 1. DFL：把 64 通道分布解码成 4 个偏移量 ==========
def dfl(position):
    """position: (n, 4*mc, h, w)。softmax 加权求和 → (n, 4, h, w)。"""
    n, c, h, w = position.shape
    mc = c // 4
    x = position.reshape(n, 4, mc, h, w).astype(np.float32)
    # 在 mc 轴上做 softmax（先减最大值防 exp 溢出）
    e = np.exp(x - x.max(axis=2, keepdims=True))
    y = e / e.sum(axis=2, keepdims=True)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(axis=2)


# ========== 2. 单头：grid/stride 还原框坐标 ==========
def box_process(position):
    """position: 单头 box 输出 (n,64,h,w)。返回 xyxy 框 (n,4,h,w)。"""
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h,
                       IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)
    position = dfl(position)
    xy = grid + 0.5 - position[:, 0:2]
    xy2 = grid + 0.5 + position[:, 2:4]
    return np.concatenate([xy * stride, xy2 * stride], axis=1)


# ========== 3. 三头：9 输出 → 标准 (1, 4+nc, 8400) ==========
def decode_three_head(input_data):
    """input_data: RKNN 9 个输出。返回 (1, 4+6, 8400)，框为 xywh。"""
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


# ========== 4. IoU：一个框对多个框 ==========
def _iou_single(box, boxes):
    """box: (4,) xyxy；boxes: (k,4)。返回 (k,) IoU。"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    w = np.clip(x2 - x1, 0, None)
    h = np.clip(y2 - y1, 0, None)
    inter = w * h
    a1 = (box[2] - box[0]) * (box[3] - box[1])
    a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a1 + a2 - inter + 1e-9)


# ========== 5. numpy NMS：多标签 + 按类抑制 ==========
def nms(pred, conf_thres=0.25, iou_thres=0.7, max_det=300):
    """pred: (1, 4+nc, N)。返回 (M, 6): xyxy, conf, cls。"""
    x = pred[0]
    if x.shape[0] < x.shape[1]:        # (10, N) → (N, 10)
        x = x.T
    boxes_xywh = x[:, :4]
    scores = x[:, 4:]

    # xywh → xyxy
    boxes = np.empty_like(boxes_xywh)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # 多标签：列出所有 (框, 类) 超过阈值的组合
    ii, jj = np.where(scores > conf_thres)
    if len(ii) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    boxes, conf = boxes[ii], scores[ii, jj]
    cls = jj.astype(np.float32)

    # 每个类内部按分数排序，逐个做 IoU 抑制
    keep = []
    for c in np.unique(cls):
        m = np.where(cls == c)[0]
        b, cf = boxes[m], conf[m]
        order = cf.argsort()[::-1]
        while len(order) > 0:
            i = order[0]
            keep.append(m[i])
            if len(keep) >= max_det:
                break
            rest = order[1:]
            if len(rest) == 0:
                break
            iou = _iou_single(b[i], b[rest])
            order = rest[iou <= iou_thres]
        if len(keep) >= max_det:
            break

    keep = np.array(keep)
    return np.concatenate(
        [boxes[keep], conf[keep, None], cls[keep, None]], axis=1)
