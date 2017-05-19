# coding: utf-8
import pickle
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader
from src.region_proposal import RegionProposalNetwork
from src.recall import recall
from src.voc_dataset_2 import *
from src.proposal import ProposalGenerator

arguments = argparse.ArgumentParser("Evaluating Region Proposal Network")
arguments.add_argument("name", help="experiment name")
arguments.add_argument("model", help="path to rpn model checkpoint")
arguments.add_argument("--dataset", help="dataset to run evalution on", choices=["train", "val"], default="train")
arguments.add_argument("-d", "--display", help="display interval", default=100, type=int)


def main():
    args = arguments.parse_args()
    name = "%s_%s" % (args.dataset, args.name)
    path = os.path.join("data/Evaluations", "recall_%s.pkl" % name)
    if os.path.exists(path):
        print("%s already exists!!" % path)
        return
    
    dataset = VOCDataSet("data", args.dataset, enabled_flip=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=5, collate_fn=collate_fn)
   
    print("[+] Bootstrapping model")
    print("[+] Loading model from %s" % args.model)
    checkpoint = torch.load(args.model)
    rpn = RegionProposalNetwork().cuda()
    softmax_m = nn.Softmax2d().cuda()
    proposal_m = ProposalGenerator()
    rpn.load_state_dict(checkpoint["state_dict"])
    
    ###################### EVALUATION ####################
    evaluate_pairs = []
    start_time = time.time()
    for i, (_, image_arr, gt, _, image_info) in enumerate(dataloader):
        image_arr = autograd.Variable(image_arr.cuda())
        logits, regressions = rpn(image_arr)
        logits = logits.resize(1, 2, logits.size(2)*9, logits.size(3))
        softmax = softmax_m(logits)
        proposals, scores = proposal_m(regressions.cpu(), softmax.cpu(), image_info[0, :])
        evaluate_pairs.append((proposals.data.numpy()[:, 1:], np.array(gt.numpy()[:, 1:], copy=True)))
        if i % args.display == (args.display - 1):
            ave_time = (time.time() - start_time) / float(args.display)
            print("[+] Processed %d images (%.3f sec/image)" % (i, ave_time))
            start_time = time.time()
    recall_result = recall(evaluate_pairs)
    with open(path, "wb") as f:
        pickle.dump(recall_result, f)
    ave_recall = sum(recall_result) / float(len(recall_result))
    print("recall[0.5]=%.3f recall_ave=%.3f" % (recall_result[0]*100., ave_recall))        

if __name__ == "__main__":
    main()
