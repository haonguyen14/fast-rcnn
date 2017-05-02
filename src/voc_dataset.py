from os import listdir
from os.path import join

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import pandas as pd
import numpy as np
import cv2
from scipy.misc import imresize

# BGR format to match VGG16 pretrained model
PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

def collate_fn(batch):
    image = batch[0][0][np.newaxis, :]
    gt_boxes = batch[0][1]
    gt_labels = batch[0][2]
    image_info = np.hstack((batch[0][0].shape[1:], batch[0][3]))
    image_name = batch[0][4]

    ret = [torch.Tensor(image), torch.Tensor(gt_boxes), torch.Tensor(image_info), None, None]
    
    if gt_labels is not None:
        ret[3] = torch.Tensor(gt_labels)
    if image_name is not None:
        ret[4] = image_name

    return ret


class VOCDataSet(Dataset):

    def __init__(self, root, image_set, size=600, include_gt_lable=False, include_difficult_gt=False, include_image_name=False):
        self._root = root
        self._image_dir = join(self._root, "JPEGImages")
        self._annotation_dir = join(self._root, "Preprocess/Annotations")
        self._size = size
        self._max_size = 1000
        
        #  get image file name
        assert (image_set == "train") | (image_set == "val"), "Invalid image set"
        with open(join(self._root, "ImageSets/Main/%s.txt" % image_set), "r") as f:
            #  remove \n character
            self._dataset_index = {i:line[0:-1] for i, line in enumerate(f)}

        self._include_gt_label = include_gt_lable
        self._include_difficult_gt = include_difficult_gt
        self._include_image_name = include_image_name

    def __getitem__(self, i):
        image_name = self._dataset_index[i]
        image_path = join(self._image_dir, "%s.jpg" % image_name)
        annotation_path = join(self._annotation_dir, "%s.jpg.csv" % self._dataset_index[i])

        image = cv2.imread(image_path).astype(np.float32)

        #  image normalization
        image = image - PIXEL_MEANS

        #  resize image
        shorter_dim = float(np.min(image.shape[0:2]))
        longer_dim = float(np.max(image.shape[0:2]))
        scale_const = float(self._size) / shorter_dim
        if (scale_const * longer_dim) > self._max_size:
            scale_const = float(self._max_size) / longer_dim
        image = cv2.resize(image, None, None, \
                fx=scale_const, fy=scale_const, \
                interpolation=cv2.INTER_LINEAR)

        #  transpose to match pytorch convolution input
        image = image.transpose(2, 0, 1)

        #  parse annotation file
        annotations = pd.read_csv(annotation_path)
        annotations = annotations.as_matrix()
        gt_idx = np.arange(0, annotations.shape[0])
        if not self._include_difficult_gt:
            gt_idx = np.where(annotations[:, 4] == 0)[0]

        bboxes = annotations[gt_idx, 0:4] * scale_const
        labels = None

        if self._include_gt_label:
            labels = annotations[gt_idx, 5]
        if not self._include_image_name:
            image_name = None

        return image, bboxes, labels, scale_const, image_name

    def __len__(self):
        return len(self._dataset_index)
