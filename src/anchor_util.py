import numpy as np

def generate_all_anchors(width, height, stride, base_anchors):

        # generating shift matrix
        shift_x = np.arange(0, width) * stride
        shift_y = np.arange(0, height) * stride

        xs, ys = np.meshgrid(shift_x, shift_y)
        xs, ys = xs.ravel(), ys.ravel()
        shifts = np.vstack([xs, ys, xs, ys])
        shifts = shifts[np.newaxis, :].transpose(2, 0, 1)
        num_anchors = base_anchors.shape[1]

        # generating all possible anchors with shape (num_anchors * w * h, 4)
        return (base_anchors + shifts).reshape(num_anchors * width * height, 4)

def generate_proposals(anchors, deltas):
    anchor_width = anchors[:, 2] - anchors[:, 0] + 1.0
    anchor_height = anchors[:, 3] - anchors[:, 1] + 1.0
    anchor_x = (anchors[:, 2] + anchors[:, 0]) * .5
    anchor_y = (anchors[:, 3] + anchors[:, 1]) * .5

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    gx = (anchor_width[:, np.newaxis] * dx) + anchor_x[:, np.newaxis]
    gy = (anchor_height[:, np.newaxis] * dy) + anchor_y[:, np.newaxis]
    gw = anchor_width[:, np.newaxis] * np.exp(dw)
    gh = anchor_height[:, np.newaxis] * np.exp(dh)
    
    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    pred_boxes[:, 0::4] = gx - (0.5 * gw)
    pred_boxes[:, 1::4] = gy - (0.5 * gh)
    pred_boxes[:, 2::4] = gx + (0.5 * gw)
    pred_boxes[:, 3::4] = gy + (0.5 * gh)

    return pred_boxes

def clip_boxes(proposals, w, h):
    proposals[:, 0::4] = np.maximum(np.minimum(proposals[:, 0::4], w-1), 0)
    proposals[:, 1::4] = np.maximum(np.minimum(proposals[:, 1::4], h-1), 0)
    proposals[:, 2::4] = np.maximum(np.minimum(proposals[:, 2::4], w-1), 0)
    proposals[:, 3::4] = np.maximum(np.minimum(proposals[:, 3::4], h-1), 0)
    return proposals

def filter_boxes(proposals, threshold):
    ws = proposals[:, 2] - proposals[:, 0] + 1
    hs = proposals[:, 3] - proposals[:, 1] + 1
    return np.where((ws >= threshold) & (hs >= threshold))[0]

def get_overlap(anchors, gt_boxes):
        def cal_area(x_min, y_min, x_max, y_max):
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            return width * height if width > 0 and height > 0 else 0.

        def iou(box1, box2):
            box1_x_min, box1_y_min, box1_x_max, box1_y_max = box_info(box1)
            box1_area = cal_area(box1_x_min, box1_y_min, box1_x_max, box1_y_max)

            box2_x_min, box2_y_min, box2_x_max, box2_y_max = box_info(box2)
            box2_area = cal_area(box2_x_min, box2_y_min, box2_x_max, box2_y_max)

            x_min = max(box1_x_min, box2_x_min)
            y_min = max(box1_y_min, box2_y_min)
            x_max = min(box1_x_max, box2_x_max)
            y_max = min(box1_y_max, box2_y_max)
            intersect_area = cal_area(x_min, y_min, x_max, y_max)

            union_area = (box1_area + box2_area) - intersect_area
            return intersect_area / union_area

        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]
        overlap = np.zeros((num_anchors, num_gt), dtype=np.float)

        for i in range(num_anchors):
            for j in range(num_gt):
                overlap[i, j] = iou(anchors[i], gt_boxes[j])

        return overlap

def box_info(box):
        return box[0], box[1], box[2], box[3]
