# coding: utf-8
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader
import numpy as np
from src.region_proposal import RegionProposalNetwork
from src.recall import recall
from src.voc_dataset import *
from src.proposal import ProposalGenerator
from os import listdir
from os.path import join

checkpoint_dir = "data/Experiments/train_wo_difficult_20/checkpoints"
checkpoints = [join(checkpoint_dir, cp) for cp in listdir(checkpoint_dir)]

cp = [None] * len(checkpoints)
for checkpoint_path in checkpoints:
    checkpoint = torch.load(checkpoint_path)
    cp[checkpoint["epoch"]] = checkpoint_path

for checkpoint in cp:
    checkpoint = torch.load(checkpoint)
    dataset = VOCDataSet("data/", "train", include_gt_lable=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=collate_fn)
    rpn = RegionProposalNetwork().cuda()
    softmax_m = nn.Softmax2d().cuda()
    rpn.load_state_dict(checkpoint["state_dict"])
    evaluate_pairs = []
    for image_arr, gt, labels in dataloader:
        image_arr = autograd.Variable(image_arr).cuda()
        logits, regressions = rpn(image_arr)
        logits = logits.resize(1, 2, logits.size(2)*9, logits.size(3))
        softmax = softmax_m(logits)
        proposal_m = ProposalGenerator()
        im_w = autograd.Variable(torch.Tensor([image_arr.size(2)]))
        im_h = autograd.Variable(torch.Tensor([image_arr.size(3)]))
        proposals, scores = proposal_m(regressions.cpu(), softmax.cpu(), im_w, im_h)
        evaluate_pairs.append((proposals.data.numpy()[:, 1:], gt.numpy()))
    print(("Epoch %d: recall=%.3f") % (checkpoint["epoch"]+1, recall(evaluate_pairs)[0] * 100.))
