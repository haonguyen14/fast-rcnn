import numpy as np
from anchor_util import *

def recall(data):
    gt_overlaps = np.zeros(0)
    num_pos = 0
    for boxes, gt_boxes in data:
        overlaps = get_overlap(boxes, gt_boxes)
        _gt_overlaps = np.zeros((gt_boxes.shape[0]))
        for i in xrange(gt_boxes.shape[0]):
            best_proposals = overlaps.argmax(axis=0)
            best_ious = overlaps.max(axis=0)
            best_gt_matched = best_ious.argmax()
            matching_proposal = best_proposals[best_gt_matched]
            _gt_overlaps[i] = overlaps[matching_proposal, best_gt_matched]
            overlaps[matching_proposal, :] = -1
            overlaps[:, best_gt_matched] = -1
        gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))
        num_pos += gt_boxes.shape[0]

    gt_overlaps = np.sort(gt_overlaps)
    thresholds = np.arange(0.5, 0.95 + 1e-5, 0.05)
    recalls = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        recalls[i] = np.sum(gt_overlaps >= t) / float(num_pos)
    
    return recalls
