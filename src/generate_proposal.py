import argparse
import os
import shutil
import pickle
import time

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader

import voc_dataset_2 as voc_dataset
import region_proposal
import proposal

arguments = argparse.ArgumentParser("Generate Proposals Using RPN")
arguments.add_argument("model", help="path to RPN checkpoint")
arguments.add_argument("-d", "--dataset", help="dataset", default="train", choices=["train", "val"])
arguments.add_argument("-p", "--path", help="proposal path", default="data/ROIs")
arguments.add_argument("--display", help="display interval", default=100, type=int)


def generate(rpn, softmax_m, proposal_generator, image_arr, image_info):
    image_arr = autograd.Variable(image_arr.cuda())
    cls_scores, bbox_preds = rpn(image_arr)

    #  reshape unnormalized classification scores to match softmax's input shape
    cls_scores = cls_scores.view(1, 2, 9 * cls_scores.size(2), cls_scores.size(3))
    cls_scores = softmax_m(cls_scores)
    proposals, scores = proposal_generator(bbox_preds.cpu(), cls_scores.cpu(), image_info)

    #  rescale proposals
    scale_factor = image_info[2]
    proposals[:, 1:].div_(scale_factor)
    return proposals.data.numpy()[:, 1:]


def main():
    global args
    args = arguments.parse_args()

    if not os.path.exists(args.model):
        print("Error: %s does not exist" % args.model)
        return

    if os.path.exists(args.path):
        print("[+] Overwriting ROIs directory")
        shutil.rmtree(args.path)
    os.makedirs(args.path)

    checkpoint = torch.load(args.model)
    train_data = voc_dataset.VOCDataSet("data", args.dataset, enabled_flip=True)
    dataloader = DataLoader(train_data, batch_size=1,
                            shuffle=True, num_workers=5, collate_fn=voc_dataset.collate_fn)

    ################### MODEL BOOTSRAP #####################
    print("[+] Bootsrapping model from %s" % args.model)
    rpn = region_proposal.RegionProposalNetwork().cuda()
    rpn.load_state_dict(checkpoint["state_dict"])
    softmax_m = nn.Softmax2d().cuda()
    proposal_generator = proposal.ProposalGenerator()

    start_time = time.time()
    for i, (indices, image_arr, _, _, image_info) in enumerate(dataloader):
        image_name = train_data.get_image_name_from_index(indices[0])
        proposals = generate(rpn, softmax_m, proposal_generator, image_arr, image_info[0, :])
        with open(os.path.join(args.path, "%s.pkl" % image_name), "wb") as f:
            pickle.dump(proposals, f)
        if i % args.display == (args.display - 1):
            ave_time = (time.time() - start_time) / float(args.display)
            print("[+] Processed %d images (%.3f sec/image)" % (i, ave_time))
            start_time = time.time()


if __name__ == "__main__":
    main()
