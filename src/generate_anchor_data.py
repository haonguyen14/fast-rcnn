import torch.nn as nn
from torch import Tensor, LongTensor
from torch.autograd import Variable
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors


class AnchorDataGenerator(nn.Module):

    def __init__(self, stride=16, include_cut_anchor=False):
        super(AnchorDataGenerator, self).__init__()

        self.ANCHOR_BASE_SIZE = 16

        self.BATCH_SIZE = 256  # number of anchors sampled
        self.FOREGROUND_ANCHOR_PERCENTAGE = 0.5
        self.BACKGROUND_ANCHOR_PERCENTAGE = 1. - self.FOREGROUND_ANCHOR_PERCENTAGE

        self.POSITIVE_IOU_THRESHOLD = 0.7
        self.NEGATIVE_IOU_THRESHOLD = 0.3

        self.BACKGROUND_LABEL = 0.
        self.FOREGROUND_LABEL = 1.
        self.IGNORE_LABEL = 2.

        self._BASE_ANCHORS = generate_anchors(self.ANCHOR_BASE_SIZE)[np.newaxis, :]
        self._NUM_ANCHORS_PER_BOX = self._BASE_ANCHORS.shape[1]
        self._include_cut_anchor = include_cut_anchor
        self._stride = stride

    def forward(self, anchor_object_scores, ground_truth_boxes, im_w, im_h):
        anchor_object_scores = anchor_object_scores.data.numpy()
        ground_truth_boxes = ground_truth_boxes.data.numpy()
        im_w, im_h = im_w.data.numpy(), im_h.data.numpy()

        height, width = anchor_object_scores.shape[2], anchor_object_scores.shape[3]

        all_anchors = self._get_all_anchors(width, height)
        num_anchors = all_anchors.shape[0]

        # get all valid anchors (those are inside the image)
        valid_anchor_indices, valid_anchors = self._get_valid_anchors(all_anchors, im_w, im_h)
        #print("[+] %d ground truth boxes in total." % len(ground_truth_boxes))
        #print("[+] Found %d valid anchors in total %d." % (len(valid_anchor_indices), num_anchors))

        # get valid anchor labels
        anchor_max_overlap_indices, valid_anchor_labels = \
            self._get_anchor_labels(valid_anchors, ground_truth_boxes)

        # ignore exceeding foreground anchors
        target_size_foregrounds = self.BATCH_SIZE * self.FOREGROUND_ANCHOR_PERCENTAGE
        foreground_indices = np.where(valid_anchor_labels == self.FOREGROUND_LABEL)[0]
        #print("[+] Need %d, found %d foreground anchor" % (target_size_foregrounds, len(foreground_indices)))
        if len(foreground_indices) > target_size_foregrounds:
            ignore_foreground_size = int(len(foreground_indices) - target_size_foregrounds)
            ignore_foregrounds = npr.choice(foreground_indices, size=ignore_foreground_size, replace=False)
            valid_anchor_labels[ignore_foregrounds] = self.IGNORE_LABEL

        # ignore exceeding background anchors
        target_size_backgrounds = self.BATCH_SIZE * self.BACKGROUND_ANCHOR_PERCENTAGE
        background_indices = np.where(valid_anchor_labels == self.BACKGROUND_LABEL)[0]
        #print("[+] Need %d, found %d background anchor" % (target_size_backgrounds, len(background_indices)))
        if len(background_indices) > target_size_backgrounds:
            ignore_background_size = int(len(background_indices) - target_size_backgrounds)
            ignore_backgrounds = npr.choice(background_indices, size=ignore_background_size, replace=False)
            valid_anchor_labels[ignore_backgrounds] = self.IGNORE_LABEL

        bbox_adjustments = self._compute_bbox_target(
            valid_anchors, ground_truth_boxes[anchor_max_overlap_indices, :])

        bbox_mask = np.zeros((len(valid_anchor_indices), 4), dtype=np.float32)
        bbox_mask[valid_anchor_labels == self.FOREGROUND_LABEL, :] = np.array([1., 1., 1., 1.])

        num_not_ignored = np.sum(valid_anchor_labels != self.IGNORE_LABEL)
        bbox_norm = np.zeros((len(valid_anchor_indices), 4), dtype=np.float32)
        bbox_norm[valid_anchor_labels == self.FOREGROUND_LABEL, :] = np.ones((1, 4)) * 1.0 / num_not_ignored

        all_anchor_labels = np.empty((num_anchors,), dtype=np.float32)
        all_anchor_labels.fill(self.IGNORE_LABEL)
        all_anchor_labels[valid_anchor_indices] = valid_anchor_labels
        all_anchor_labels = all_anchor_labels.reshape(1, height, width, self._NUM_ANCHORS_PER_BOX)
        all_anchor_labels = all_anchor_labels.transpose(0, 3, 1, 2)
        all_anchor_labels = all_anchor_labels.reshape(1, 1, self._NUM_ANCHORS_PER_BOX * height, width)

        all_bbox_adjustments = np.empty((num_anchors, 4), dtype=np.float32)
        all_bbox_adjustments.fill(0.)
        all_bbox_adjustments[valid_anchor_indices, :] = bbox_adjustments
        all_bbox_adjustments = all_bbox_adjustments.reshape(1, height, width, 4 * self._NUM_ANCHORS_PER_BOX)
        all_bbox_adjustments = all_bbox_adjustments.transpose(0, 3, 1, 2)

        all_bbox_adjustment_weights = np.empty((num_anchors, 4), dtype=np.float32)
        all_bbox_adjustment_weights.fill(0.)
        all_bbox_adjustment_weights[valid_anchor_indices, :] = bbox_mask * bbox_norm
        all_bbox_adjustment_weights = \
            all_bbox_adjustment_weights.reshape(1, height, width, 4 * self._NUM_ANCHORS_PER_BOX)
        all_bbox_adjustment_weights = all_bbox_adjustment_weights.transpose(0, 3, 1, 2)

        return (
            Variable(LongTensor(all_anchor_labels.astype(np.long))),
            Variable(Tensor(all_bbox_adjustments)),
            Variable(Tensor(all_bbox_adjustment_weights))
        )

    def _get_valid_anchors(self, anchors, im_w, im_h):
        if self._include_cut_anchor:
            filter_indices = np.arange(0, anchors.shape[0])
        else:
            filter_indices = np.where(
                (anchors[:, 0] >= 0.) & (anchors[:, 1] >= 0.) &
                (anchors[:, 2] < im_w) & (anchors[:, 3] < im_h)
            )[0]

        return filter_indices, anchors[filter_indices]

    def _box_info(self, box):
        return box[0], box[1], box[2], box[3]

    def _get_overlap(self, anchors, gt_boxes):

        def cal_area(x_min, y_min, x_max, y_max):
            width = x_max - x_min + 1
            height =  y_max - y_min + 1
            return width * height if width > 0 and height > 0 else 0.

        def iou(box1, box2):
            box1_x_min, box1_y_min, box1_x_max, box1_y_max = self._box_info(box1)
            box1_area = cal_area(box1_x_min, box1_y_min, box1_x_max, box1_y_max)

            box2_x_min, box2_y_min, box2_x_max, box2_y_max = self._box_info(box2)
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

    def _get_anchor_labels(self, anchors, gt_boxes):
        overlap = self._get_overlap(anchors, gt_boxes)

        # for each anchor what is its max IOU with ground truth boxes
        anchor_max_overlap_indices = np.argmax(overlap, axis=1)
        anchor_max_overlaps = overlap[np.arange(anchors.shape[0]), anchor_max_overlap_indices]

        # for each ground truth box what anchors maximize their IOU
        gt_max_overlaps = overlap[np.argmax(overlap, axis=0), np.arange(overlap.shape[1])]
        gt_max_overlap_indices = np.where(overlap == gt_max_overlaps)[0]

        labels = np.empty((anchors.shape[0], ), dtype=np.float32)
        labels.fill(self.IGNORE_LABEL)

        labels[anchor_max_overlaps < self.NEGATIVE_IOU_THRESHOLD] = self.BACKGROUND_LABEL
        labels[gt_max_overlap_indices] = self.FOREGROUND_LABEL
        labels[anchor_max_overlaps >= self.POSITIVE_IOU_THRESHOLD] = self.FOREGROUND_LABEL

        return anchor_max_overlap_indices, labels

    def _compute_bbox_target(self, anchors, gt_boxes):
        anchor_width = anchors[:, 2] - anchors[:, 0] + 1.0
        anchor_height = anchors[:, 3] - anchors[:, 1] + 1.0
        anchor_x = (anchors[:, 2] + anchors[:, 0]) * .5
        anchor_y = (anchors[:, 3] + anchors[:, 1]) * .5

        gt_width = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
        gt_height = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
        gt_x = (gt_boxes[:, 2] + gt_boxes[:, 0]) * .5
        gt_y = (gt_boxes[:, 3] + gt_boxes[:, 1]) * .5

        dx = (gt_x - anchor_x) / anchor_width
        dy = (gt_y - anchor_y) / anchor_height
        dw = np.log(gt_width / anchor_width)
        dh = np.log(gt_height / anchor_height)

        return np.vstack([dx, dy, dw, dh]).transpose()

    def _get_all_anchors(self, width, height):

        # generating shift matrix
        shift_x = np.arange(0, width) * self._stride
        shift_y = np.arange(0, height) * self._stride

        xs, ys = np.meshgrid(shift_x, shift_y)
        xs, ys = xs.ravel(), ys.ravel()
        shifts = np.vstack([xs, ys, xs, ys])
        shifts = shifts[np.newaxis, :].transpose(2, 0, 1)

        # generating all possible anchors with shape (num_anchors * w * h, 4)
        return (self._BASE_ANCHORS + shifts).reshape(self._NUM_ANCHORS_PER_BOX * width * height, 4)
