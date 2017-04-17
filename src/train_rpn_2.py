from os import listdir, makedirs
from os.path import join, exists

import torch
import torch.optim as opt
from torch.nn import SmoothL1Loss
from torch.autograd import Variable
from torch.utils.data import DataLoader

from voc_dataset import VOCDataSet, collate_fn
from region_proposal import RegionProposalNetwork, get_target_weights
from generate_anchor_data import AnchorDataGenerator

import sys
import pickle
import time

DATA_PATH = "data/Experiments"

def setup_experiment_dir(experiment):
    experiment_dir = join(DATA_PATH, experiment)
    checkpoint_dir = join(experiment_dir, "checkpoints")
    if exists(experiment_dir):
        raise Exception("Experiment %s is already existed" % experiment)

    makedirs(experiment_dir)
    makedirs(checkpoint_dir)

    return {
        "main": experiment_dir,
        "checkpoint": checkpoint_dir
    }

if __name__ == "__main__":
    num_epoches = int(sys.argv[2])

    print("[+] Setting up experiment environments...")
    experiment = setup_experiment_dir(sys.argv[1])

    print("[+] Initializing model...")
    rpn = RegionProposalNetwork()
    rpn = rpn.cuda()

    anchor_generator = AnchorDataGenerator()

    #  classification loss
    log_softmax_func = torch.nn.LogSoftmax()
    log_softmax_func = log_softmax_func.cuda()

    nll_loss_func = torch.nn.NLLLoss2d(weight=get_target_weights(), size_average=False)
    nll_loss_func = nll_loss_func.cuda()

    #  regression loss 
    regression_loss_func = SmoothL1Loss(size_average=False)
    regression_loss_func = regression_loss_func.cuda()

    optimizer = opt.SGD(
        rpn.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005,
    )

    train_data = VOCDataSet("data", "train")
    dataloader = DataLoader(train_data,
                            batch_size=1,
                            shuffle=True,
                            num_workers=5,
                            collate_fn=collate_fn 
                           )

    loss_in_epoch = []
    loss_epoches = []
    log_interval = 500

    print("[+] Start training...")
    for epoch in range(num_epoches):
        
        start_time = time.time()
        running_loss = 0.0
        loss_epoch = 0.0
        counter = 0

        for i, (image, ground_truth_boxes, _) in enumerate(dataloader):
            image = Variable(torch.Tensor(image).cuda())
            im_h = Variable(torch.Tensor([image.size(2)]))
            im_w = Variable(torch.Tensor([image.size(3)]))
            ground_truth_boxes = Variable(torch.Tensor(ground_truth_boxes))

            optimizer.zero_grad()

            logits, regressions = rpn(image)
            width = Variable(torch.Tensor([logits.size(3)]))
            height = Variable(torch.Tensor([logits.size(2)]))

            labels, bbox_targets, bbox_inside_weights, regression_norm = anchor_generator(
                width, height, ground_truth_boxes, im_w, im_h)

            bbox_targets = bbox_targets.cuda()
            bbox_inside_weights = bbox_inside_weights.cuda()
            regression_norm = regression_norm.cuda()

            #  calculate negative log loss
            #  TODO: pull number of anchors per box out to a configuration
            logits = logits.view(1, 2, -1, logits.size(3))
            labels = labels.view(1, labels.size(2), labels.size(3))
            labels = labels.cuda()

            log_softmax = log_softmax_func(logits)
            log_softmax = torch.cat(
                (
                    log_softmax,
                    Variable(torch.zeros([1, 1, log_softmax.size(2), log_softmax.size(3)]).cuda())
                ), 1)  # add an additional layer so that we can have 3 classes with 1 ignored

            nll_loss = nll_loss_func(log_softmax, labels)

            #  calculate regression loss
            regressions = torch.mul(regressions, bbox_inside_weights)
            bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
            regression_loss = regression_loss_func(regressions, bbox_targets) * (regression_norm + 1e-4)

            #  TODO: pull 256 to a configuration for batch size
            loss = (nll_loss / 256.) + regression_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            loss_epoch += loss.data[0]
           
            if i % log_interval == (log_interval-1):
                loss_in_epoch.append(running_loss / log_interval)
                running_loss = 0.0

            counter += 1

        total_time = time.time() - start_time
        ave_loss = loss_epoch / float(counter)
        print("Epoch %d/%d: ave_loss=%.3f total_time=%.3f" % (epoch+1, num_epoches, ave_loss, total_time))
    
        torch.save({
            "epoch": epoch,
            "ave_loss": ave_loss,
            "running_loss": running_loss,
            "state_dict": rpn.state_dict()
        }, join(experiment["checkpoint"], "checkpoint_%d.tar" % epoch))
