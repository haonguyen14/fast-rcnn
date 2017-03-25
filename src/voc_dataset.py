from os import listdir
from os.path import join

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import pandas as pd
import numpy as np
from PIL import Image


PIXEL_MEANS = np.array([[[0.48462227599918]],
                          [[0.45624044862054]],
                          [[0.40588363755159]]])

PIXEL_STDS = np.array([[[0.22889466674951]],
                        [[0.22446679341259]],
                        [[0.22495548344775]]])

def collate_fn(batch):
    image = batch[0][0][np.newaxis, :]
    gt_boxes = batch[0][1]
    return torch.Tensor(image), torch.Tensor(gt_boxes)

class VOCDataSet(Dataset):

    def __init__(self, root, image_set, include_gt_lable=False, download=False):
        self._root = root
        self._image_dir = join(self._root, "JPEGImages")
        self._annotation_dir = join(self._root, "Preprocess/Annotations")
        
        #  get image file name
        assert (image_set == "train") | (image_set == "val"), "Invalid image set"
        with open(join(self._root, "ImageSets/Main/%s.txt" % image_set), "r") as f:
            #  remove \n character
            self._dataset_index = {i:line[0:-1] for i, line in enumerate(f)}

        self._include_gt_label = include_gt_lable

    def __getitem__(self, i):
        image_path = join(self._image_dir, "%s.jpg" % self._dataset_index[i])
        annotation_path = join(self._annotation_dir, "%s.jpg.csv" % self._dataset_index[i])

        image = Image.open(image_path).convert("RGB")
        image = np.asarray(image).astype(np.float).transpose(2, 0, 1)

        #  image normalization
        image /= 255.0
        image = (image - PIXEL_MEANS) / PIXEL_STDS

        #  parse annotation file
        annotations = pd.read_csv(annotation_path)
        annotations = annotations.as_matrix()
        if not self._include_gt_label:
            annotations = annotations[:, 0:4]

        return image, annotations

    def __len__(self):
        return len(self._dataset_index)
