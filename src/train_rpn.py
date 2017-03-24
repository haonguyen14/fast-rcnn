from os import listdir
from os.path import join

import torch
from torch.autograd import Variable

import numpy as np
import numpy.random as npr
import pandas as pd
from PIL import Image

from region_proposal import RegionProposalNetwork, get_target_weights
from smooth_l1_loss import SmoothL1LossFunc
from generate_anchor_data import AnchorDataGenerator


EPOCH = 1

DATA_DIR = "data"
IMAGE_DIR = join(DATA_DIR, "JPEGImages")
ANNOTATION_DIR = join(DATA_DIR, "Preprocess/Annotations")

def get_image_input(image_path):
    image = Image.open(image_path)
    image_a = np.asarray(image).astype(np.float32).transpose(2, 0, 1)
    image_a = image_a[np.newaxis, :]

    input = Variable(torch.Tensor(image_a))
    im_h = Variable(torch.Tensor([image_a.shape[2]]))
    im_w = Variable(torch.Tensor([image_a.shape[3]]))

    return input, im_w, im_h

def get_ground_truth_boxes(annotation_path):
    df = pd.read_csv(annotation_path)
    gt = df.as_matrix()
    gt = gt[:, 0:4]
    return Variable(torch.Tensor(gt))

if __name__ == "__main__":
    image_files = [filename for filename in listdir(IMAGE_DIR)]

    image_paths = [
        (join(IMAGE_DIR, filename), join(ANNOTATION_DIR, filename + ".csv"))
        for filename in image_files]

    rpn = RegionProposalNetwork()
    anchor_generator = AnchorDataGenerator()
    nll_loss_func = torch.nn.CrossEntropyLoss(weight=get_target_weights())
    regression_loss_func = SmoothL1LossFunc()

    for epoch in range(EPOCH):
        indices = np.arange(len(image_paths))
        npr.shuffle(indices)
        indices = indices.tolist()
       
        for i in indices:
            image_path, annotation_path = image_paths[i]
            image, im_w, im_h = get_image_input(image_path)
            ground_truth_boxes = get_ground_truth_boxes(annotation_path)

            logits, regressions = rpn(image)
            labels, bbox_targets, bbox_weights = anchor_generator(
                logits, ground_truth_boxes, im_w, im_h)

            nll_loss = nll_loss_func()
            regression_loss = regression_loss_func(regressions, bbox_targets, bbox_weights)
