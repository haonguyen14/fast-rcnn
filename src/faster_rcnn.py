import torch
import torch.nn as nn
import torchvision.models as models
import roi_pooling.roi_pooling as roi_pooling
import region_proposal
from collections import OrderedDict


class FasterRCNN(nn.Module):

    def __init__(self, load_stage_2=None):
        super(FasterRCNN, self).__init__()

        #  VGG16 feature map
        if load_stage_2 is None:
            vgg = models.vgg16(pretrained=True)
            self._features = self._build_vgg_layers(vgg)
            for m in list(self._features.children())[:10]:
                self.set_requires_grad(m, False)
        else:
            self._features = self._get_feature_map_rpn(torch.load(load_stage_2)["state_dict"])
            for m in list(self._features.children()):
                self.set_requires_grad(m, False)

        #  ROI pooling with spacial scale (1. / 16.) ~ 0.0625
        self._roi_pooling = roi_pooling.ROIPooling((7, 7), spatial_scale=0.0625)

        # Fully-connected layers
        self._fc1 = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5, inplace=False)
        )
        self._fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5, inplace=False)
        )

        #  Classification and regression heads
        self._cls_score = nn.Linear(4096, 21)
        self._bbox_pred = nn.Linear(4096, 84)

        #  Initialize weights
        self._initialize_weights(self._cls_score, std=0.01)
        self._initialize_weights(self._bbox_pred, std=0.001)

    def forward(self, images, rois):
        x = self._features(images)
        x = self._roi_pooling(x, rois)
        x = x.view(x.size(0), -1)
        x = self._fc1(x)
        x = self._fc2(x)
        return self._cls_score(x), self._bbox_pred(x)

    def get_feature_layer(self):
        return self._features

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

    def _initialize_weights(self, module, std=0.01):
        for m in module.modules():
            if isinstance(m, nn.Conv2d) | isinstance(m, nn.Linear):
                m.weight.data.normal_(0, std)
                if m.bias is not None:
                    m.bias.data.zero_()

    def set_requires_grad(self, module, requires_grad):
        for params in module.parameters():
            params.requires_grad = requires_grad

    def _get_feature_map_rpn(self, state_dict):
        rpn = region_proposal.RegionProposalNetwork()
        rpn.load_state_dict(state_dict)
        return rpn.get_feature_layer()
