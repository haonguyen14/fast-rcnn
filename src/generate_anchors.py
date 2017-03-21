import numpy as np


def get_box_info(box):

    width = box[2] - box[0] + 1
    height = box[3] - box[1] + 1
    x_center = (box[2] + box[0]) / 2.0
    y_center = (box[3] + box[1]) / 2.0
    return width, height, x_center, y_center


def make_anchors(ws, hs, x_center, y_center):

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]

    return np.hstack((
        x_center - 0.5 * (ws - 1),
        y_center - 0.5 * (hs - 1),
        x_center + 0.5 * (ws - 1),
        y_center + 0.5 * (hs - 1)
    ))


def generate_ratios_anchors(base_box, ratios):

    base_box_width, base_box_height, x_center, y_center = get_box_info(base_box)
    base_box_area = base_box_width * base_box_height

    ws = np.round(np.sqrt(base_box_area / ratios))  # w = sqrt(area/ratio)
    hs = np.round(ws * ratios)  # h = ratio * w

    return make_anchors(ws, hs, x_center, y_center)


def generate_scale_anchors(anchor, scales):

    base_box_width, base_box_height, x_center, y_center = get_box_info(anchor)
    hs = base_box_height * scales
    ws = base_box_width * scales

    return make_anchors(ws, hs, x_center, y_center)


def generate_anchors(base_size, ratios=np.array([0.5, 1, 2]), scales=2**np.arange(3, 6)):

    base_box = np.array([0, 0, base_size-1, base_size-1])
    ratio_anchors = generate_ratios_anchors(base_box, ratios)
    scale_anchors = np.vstack([
                                  generate_scale_anchors(ratio_anchors[i, :], scales)
                                  for i in range(ratio_anchors.shape[0])
                                  ])

    return scale_anchors
