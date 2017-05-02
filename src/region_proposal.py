from collections import OrderedDict

import torch
import torch.nn as nn
import torch.autograd as autograd
import torchvision.models as models

import math
import numpy as np


IGNORE_LABEL = 2


class RegionProposalNetwork(nn.Module):

    def __init__(self, pretrained=True):
        super(RegionProposalNetwork, self).__init__()

        #  VGG16 feature map
        vgg = models.vgg16(pretrained=pretrained)
        self._convolutions = self._build_vgg_layers(vgg)
        for m in list(self._convolutions.children())[:10]:
            self.set_requires_grad(m, False)

        self._conv_3_512 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self._relu_3 = nn.ReLU(inplace=True)
        self._initialize_convo_weights(self._conv_3_512)

        self._spartial_linear_objectness = nn.Conv2d(512, 18, 1, stride=1, padding=0)
        self._initialize_convo_weights(self._spartial_linear_objectness)

        self._spartial_linear_bbox_reg = nn.Conv2d(512, 36, 1, stride=1, padding=0)
        self._initialize_convo_weights(self._spartial_linear_bbox_reg)

    def forward(self, x):
        x = self._convolutions(x)
        x = self._conv_3_512(x)
        x = self._relu_3(x)
        objectness_logits = self._spartial_linear_objectness(x)
        bbox_regressors = self._spartial_linear_bbox_reg(x)
        return objectness_logits, bbox_regressors

    def _build_vgg_layers(self, vgg):
        features = list(vgg.features.children())[:-1]

        # build OrderedDict
        curr_layer_idx = 1
        curr_idx = 1
        modules = []
        for f in features:
            if isinstance(f, nn.Conv2d):
                name = "Conv%d_%d" % (curr_layer_idx, curr_idx)
            elif isinstance(f, nn.ReLU):
                name = "Relu%d_%d" % (curr_layer_idx, curr_idx)
                curr_idx += 1
            elif isinstance(f, nn.MaxPool2d):
                name = "MaxPool%d" % curr_layer_idx
                curr_layer_idx += 1
                curr_idx = 1
            else:
                raise Exception("Unexpected layer")
            modules.append((name, f))

        return nn.Sequential(OrderedDict(modules))

    def _initialize_convo_weights(self, m):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

    def set_requires_grad(self, module, requires_grad):
        for params in module.parameters():
            params.requires_grad = requires_grad

def get_target_weights():

    weights = np.ones(3)
    weights[IGNORE_LABEL] = 0.0
    return torch.Tensor(weights)
