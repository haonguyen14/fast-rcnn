# coding: utf-8
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader
import numpy as np
from src.region_proposal import RegionProposalNetwork
from src.recall import recall
from src.voc_dataset_2 import *
from src.proposal import ProposalGenerator
from os import listdir
from os.path import join

checkpoint_dir = "data/Experiments/rpn_stage_2/checkpoints"
checkpoints = [join(checkpoint_dir, cp) for cp in listdir(checkpoint_dir)]

cp = [None] * len(checkpoints)
for checkpoint_path in checkpoints:
    checkpoint = torch.load(checkpoint_path)
    cp[checkpoint["epoch"]] = checkpoint_path

train_data = VOCDataSet("data", "train")
dataloader = DataLoader(train_data,
                        batch_size=1,
                        shuffle=True,
                        num_workers=5,
                        collate_fn=collate_fn
                        )

for checkpoint_path in cp:
    checkpoint = torch.load(checkpoint_path)
    rpn = RegionProposalNetwork().cuda()
    softmax_m = nn.Softmax2d().cuda()
    rpn.load_state_dict(checkpoint["state_dict"])
    evaluate_pairs = []
    for _, image_arr, gt, _, image_info in dataloader:
        image_arr = autograd.Variable(image_arr).cuda()
        logits, regressions = rpn(image_arr)
        logits = logits.resize(1, 2, logits.size(2)*9, logits.size(3))
        softmax = softmax_m(logits)
        proposal_m = ProposalGenerator()
        proposals, scores = proposal_m(regressions.cpu(), softmax.cpu(), image_info[0, :])
        evaluate_pairs.append((proposals.data.numpy()[:, 1:], np.array(gt.numpy()[:, 1:], copy=True)))
    recall_result = recall(evaluate_pairs) 
    print(("Epoch %d: recall[0.5]=%.3f recall_ave=%.3f") % (checkpoint["epoch"]+1, recall_result[0]*100., sum(recall_result)/float(len(recall_result))))
