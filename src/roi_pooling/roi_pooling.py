import torch
import torch.nn as nn

FG_FRACTION = 0.25

class ROIPooling(nn.Module):

    def __init__(self, pooled_shape, spatial_scale):
        super(ROIPooling, self).__init__()
        assert len(pooled_shape) == 2
        self._pooled_shape = pooled_shape
        self._spatial_scale = spatial_scale
        self._adaptive_max_pooling = nn.AdaptiveMaxPool2d(pooled_shape)

    def forward(self, features, rois):
        """
        ROI pooling using R ROIs in a batch of images
        :param features: convolution features of shape (batch_size, output_channels, height, width)
        :param rois: a long tensor of shape (num_rois, 5)
        :return: pooled features of shape (num_rois, output_channels, pooled_height, pooled_width)
        """
        assert features.dim() == 4
        assert rois.dim() == 2 and rois.size(1) == 5

        num_rois = rois.size(0)
        rois = rois.data.float()
        rois[:, 1:].mul_(self._spatial_scale)
        rois = rois.long()

        feature_pooled = []
        for i in xrange(num_rois):
            roi = rois[i, :]
            roi_patch = features.narrow(0, roi[0], 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
            feature_pooled.append(self._adaptive_max_pooling(roi_patch))

        ret = torch.cat(feature_pooled, 0)
        assert ret.dim() == 4
        assert ret.size(0) == num_rois and ret.size(1) == features.size(1)
        assert ret.size(2) == self._pooled_shape[0] and ret.size(3) == self._pooled_shape[1]
        return ret
