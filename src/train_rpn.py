from os import listdir
from os.path import join

import torch
import torch.optim as opt
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import numpy.random as npr
import pandas as pd
from PIL import Image

from voc_dataset import VOCDataSet, collate_fn
from region_proposal import RegionProposalNetwork, get_target_weights
from ext.smooth_l1_loss import SmoothL1Loss
from generate_anchor_data import AnchorDataGenerator


EPOCH = 1

if __name__ == "__main__":
    rpn = RegionProposalNetwork()
    rpn = rpn.cuda()

    anchor_generator = AnchorDataGenerator()

    #  classification loss
    log_softmax_func = torch.nn.LogSoftmax()
    log_softmax_func = log_softmax_func.cuda()

    nll_loss_func = torch.nn.NLLLoss2d(weight=get_target_weights(), size_average=False)
    nll_loss_func = nll_loss_func.cuda()

    #  regression loss 
    regression_loss_func = SmoothL1Loss()
    regression_loss_func = regression_loss_func.cuda()

    optimizer = opt.SGD(rpn.parameters(), lr=0.001, momentum=0.9)

    train_data = VOCDataSet("data", "train")
    dataloader = DataLoader(train_data,
                            batch_size=1,
                            shuffle=True,
                            num_workers=5,
                            collate_fn=collate_fn 
                           )

    for epoch in range(EPOCH):
        
        running_loss = 0.0

        for i, (image, ground_truth_boxes, _) in enumerate(dataloader):
            image = Variable(torch.Tensor(image).cuda())
            im_h = Variable(torch.Tensor([image.size(2)]))
            im_w = Variable(torch.Tensor([image.size(3)]))
            ground_truth_boxes = Variable(torch.Tensor(ground_truth_boxes))

            optimizer.zero_grad()

            logits, regressions = rpn(image)
            width = Variable(torch.Tensor([logits.size(3)]))
            height = Variable(torch.Tensor([logits.size(2)]))

            labels, bbox_targets, bbox_weights = anchor_generator(
                width, height, ground_truth_boxes, im_w, im_h)

            bbox_targets = bbox_targets.cuda()
            bbox_weights = bbox_weights.cuda()

            #  calculate negative log loss
            #  TODO: pull number of anchors per box out to a configuration
            logits = logits.resize(1, 2, 9 * logits.size(2), logits.size(3))
            labels = labels.resize(1, labels.size(2), labels.size(3))
            labels = labels.cuda()

            log_softmax = log_softmax_func(logits)
            log_softmax = torch.cat(
                (
                    log_softmax,
                    Variable(torch.zeros([1, 1, log_softmax.size(2), log_softmax.size(3)]).cuda())
                ), 1)  # add an additional layer so that we can have 3 classes with 1 ignored

            nll_loss = nll_loss_func(log_softmax, labels)

            #  calculate regression loss
            regression_loss = regression_loss_func(regressions, bbox_targets, bbox_weights)

            #  TODO: pull 256 to a configuration for batch size
            loss = (nll_loss / 256.) + regression_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
           
            """
            if i % 1 == 0:
                print("[+] %d batches, loss = %.3f" % (i, running_loss / 1.))
                running_loss = 0.0
            """
