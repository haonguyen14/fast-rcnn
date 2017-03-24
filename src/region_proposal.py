import torch
import torch.nn as nn
import torch.autograd as autograd
import torchvision.models as models

import math
import numpy as np


IGNORE_LABEL = 2


class RegionProposalNetwork(nn.Module):

    def __init__(self):

        super(RegionProposalNetwork, self).__init__()

        self._convolutions = models.vgg.make_layers(models.vgg.cfg['D'])
        self._conv_3_256 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self._spartial_linear_objectness = nn.Conv2d(256, 18, 1, stride=1, padding=0)
        self._spartial_linear_bbox_reg = nn.Conv2d(256, 36, 1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x):

        x = self._convolutions(x)
        x = self._conv_3_256(x)

        height, width = x.size(2), x.size(3)

        objectness_logits = self._spartial_linear_objectness(x)
        bbox_regressors = self._spartial_linear_bbox_reg(x)

        return objectness_logits, bbox_regressors

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                num_weights = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / num_weights))
                if m.bias is not None:
                    m.bias.data.zero_()

def get_target_weights():

    weights = np.ones(3)
    weights[IGNORE_LABEL] = 0.0
    return torch.Tensor(weights)
