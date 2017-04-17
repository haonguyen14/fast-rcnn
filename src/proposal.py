import torch
import torch.autograd as autograd
import torch.nn as nn

import numpy as np

from generate_anchors import generate_anchors
from anchor_util import *
from nms import py_cpu_nms

class ProposalGenerator(nn.Module):

    def __init__(self, stride=16):
        super(ProposalGenerator, self).__init__()

        self.ANCHOR_BASE_SIZE = 16
        self._STRIDE = stride
        self._BASE_ANCHORS = generate_anchors(self.ANCHOR_BASE_SIZE)[np.newaxis, :]
        self._NUM_ANCHORS_PER_BOX = self._BASE_ANCHORS.shape[1]
    
    def forward(self, bbox_deltas, scores, im_w, im_h):
        scores = scores.data.numpy()
        scores = scores.reshape([1, 18, scores.shape[2]/9, scores.shape[3]])
        scores = scores[:, 9:, :, :]
        bbox_deltas = bbox_deltas.data.numpy()

        height = scores.shape[2]
        width = scores.shape[3]
        anchors = generate_all_anchors(
            width,
            height,
            self._STRIDE,
            self._BASE_ANCHORS
        )

        scores = scores.transpose([0, 2, 3, 1]).reshape([-1, 1])
        bbox_deltas = bbox_deltas.transpose([0, 2, 3, 1]).reshape([-1, 4])
        proposals = generate_proposals(anchors, bbox_deltas)
        proposals = clip_boxes(proposals, im_w, im_h)
        
        threshold = 16. * (im_h.data[0] / im_w.data[0])
        keep = filter_boxes(proposals, threshold)
        proposals = proposals[keep, :]
        scores = scores[keep]

        order = scores.ravel().argsort()[::-1]
        order = order[:6000]
        proposals = proposals[order, :]
        scores = scores[order]

        keep = py_cpu_nms(np.hstack((proposals, scores)), 0.7) 
        keep = keep[:2000]
        proposals = proposals[keep, :]
        scores = scores[keep]

        batch_idx = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_idx, proposals.astype(np.float32, copy=False)))

        return autograd.Variable(torch.Tensor(blob)), \
                autograd.Variable(torch.Tensor(scores))
