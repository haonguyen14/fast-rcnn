import argparse

from os import makedirs
from os.path import join, exists
from shutil import rmtree

import torch
import torch.optim as opt
from torch.nn import SmoothL1Loss, L1Loss
from torch.autograd import Variable
from torch.utils.data import DataLoader

from voc_dataset_2 import VOCDataSet, collate_fn
from region_proposal import RegionProposalNetwork, get_target_weights
from generate_anchor_data import AnchorDataGenerator
from average import AverageMeter

import time

arguments = argparse.ArgumentParser(description="Training Region Proposal Network")
arguments.add_argument("name", help="experiment name")
arguments.add_argument("epoch", help="number of epoches", type=int)
arguments.add_argument("--stage-1-path", help="path to stage 1 checkpoint", default=None)
arguments.add_argument("--batch-size", help="batch size (number of anchors per image)", type=int, default=256)
arguments.add_argument("-e", "--experiments", help="experiment path", default="data/Experiments")
arguments.add_argument("-d", "--display", help="display interval", type=int, default=500)
arguments.add_argument("-r", "--resume", help="resume from a checkpoint", default=None)
arguments.add_argument("--lr", help="base learning rate", type=float, default=0.001)
arguments.add_argument("--momentum", help="momentum constant", type=float, default=0.9)
arguments.add_argument("--gamma", help="learning rate decay constant", type=float, default=0.1)
arguments.add_argument("--lr-stepsize", help="learning rate decay stepsize", type=int, default=8)
arguments.add_argument("--replace", help="replace previous experiment with similar name", action="store_true")


def create_experiment_dir():
    experiment_dir = join(args.experiments, args.name)
    checkpoint_dir = join(experiment_dir, "checkpoints")

    if exists(experiment_dir) and args.replace:
        rmtree(experiment_dir)
    if exists(experiment_dir):
        raise Exception("Experiment %s is already existed" % args.name)

    makedirs(experiment_dir)
    makedirs(checkpoint_dir)

    return {
        "main": experiment_dir,
        "checkpoint": checkpoint_dir
    }

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (args.gamma ** (epoch // args.lr_stepsize))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def main():
    global args
    args = arguments.parse_args()
    experiment_env = create_experiment_dir()

    train_data = VOCDataSet("data", "train")
    dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=5, collate_fn=collate_fn)

    ################### MODEL BOOTSRAP #####################
    print("[+] Bootstrapping model")
    if args.stage_1_path is not None:
        print("[+] Loading stage 1 weights")
    rpn = RegionProposalNetwork(load_stage_1=args.stage_1_path).cuda()

    if args.resume is not None:
        print("[+] Resuming from %s" % args.resume)
        checkpoint = torch.load(args.resume)
        rpn.load_state_dict(checkpoint["state_dict"])

    anchor_generator = AnchorDataGenerator()
    log_softmax_func = torch.nn.LogSoftmax().cuda()
    nll_loss_func = torch.nn.NLLLoss2d(weight=get_target_weights(), size_average=False).cuda()
    regression_loss_func = L1Loss(size_average=False).cuda()
    optimizer = opt.SGD([params for params in rpn.parameters() if params.requires_grad],
                        lr=args.lr, momentum=args.momentum, weight_decay=0.0005)

    ################### MODEL TRAINING #####################
    print("[+] Training model")
    start_epoch = 0 if args.resume is None else checkpoint["epoch"]
    for epoch in xrange(start_epoch, args.epoch):
        adjust_learning_rate(optimizer, epoch)

        train(experiment_env, dataloader,
              rpn, anchor_generator,
              log_softmax_func, nll_loss_func,
              regression_loss_func,
              optimizer,
              epoch)


def train(experiment,
          dataloader,
          rpn, anchor_generator,
          log_softmax_func, nll_loss_func, regression_loss_func, optimizer, epoch):

    epoch_start_time = time.time()
    epoch_loss = AverageMeter()

    for i, (_, image, ground_truth_boxes, _, _) in enumerate(dataloader):
        image = Variable(torch.Tensor(image).cuda())
        im_h = Variable(torch.Tensor([image.size(2)]))
        im_w = Variable(torch.Tensor([image.size(3)]))
        ground_truth_boxes = Variable(torch.Tensor(ground_truth_boxes[:, 1:]))

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

        loss = (nll_loss / float(args.batch_size)) + regression_loss
        loss.backward()
        optimizer.step()

        epoch_loss.update(loss.data[0])

        if i % args.display == (args.display - 1):
            print("Epoch %d/%d/%d: ave_loss=%.3f" % (i, epoch + 1, args.epoch, epoch_loss.ave))

            # for params in rpn.parameters():
            #     param_np = params.cpu().data.numpy()
            #     if np.isnan(param_np).any():
            #         print(param_np)
            #         assert False, "NaN found!!"

    total_time = epoch_start_time - time.time()
    print("Epoch %d/%d: ave_loss=%.3f total_time=%.3f" % (epoch + 1, args.epoch, epoch_loss.ave, total_time))
    torch.save({
        "epoch": epoch,
        "ave_loss": epoch_loss.ave,
        "state_dict": rpn.state_dict()
    }, join(experiment["checkpoint"], "checkpoint_%d.tar" % epoch))

if __name__ == "__main__":
    main()
