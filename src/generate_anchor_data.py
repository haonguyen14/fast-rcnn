import torch
import torch.nn as nn
import numpy as np
from generate_anchors import generate_anchors


class AnchorDataGenerator(nn.Module):

    def __init__(self, stride=16, include_cut_anchor=False):
        super(AnchorDataGenerator, self).__init__()
        
        ANCHOR_BASE_SIZE = 16
        
        self._BASE_ANCHORS = generate_anchors(ANCHOR_BASE_SIZE)[np.newaxis, :]
        self._include_cut_anchor = include_cut_anchor
        self._stride = stride

    def forward(self, anchor_object_scores, ground_truth_boxes):

        height, width = anchor_object_scores.size(2), anchor_object_scores.size(3)

        # generating shift matrix
        shift_x = np.arange(0, width) * seft._stride
        shift_y = np.arange(0, height) * seft._stride

        xs, ys = np.meshgrid(shift_x, shift_y)
        xs, ys = xs.ravel(), ys.ravel()
        shifts = np.vstack([xs, ys, xs, ys])
        shifts = shifts[np.newaxis, :].transpose(2, 0, 1)

        # generating all possible anchors with shape (w*h, num_anchors, 4)
        all_anchors = self._BASE_ANCHORS + shifts
