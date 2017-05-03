import argparse

from os import makedirs
from os.path import join, exists
from shutil import rmtree

import torch
import torch.optim as opt
from torch.nn import SmoothL1Loss, CrossEntropyLoss
from torch.autograd import Variable
from torch.utils.data import DataLoader

from voc_dataset_2 import VOCDataSetROIs, collate_rois_fn
from faster_rcnn import FasterRCNN
from average import AverageMeter

import time

arguments = argparse.ArgumentParser(description="Training Fast-RCNN")
arguments.add_argument("name", help="experiment name")
arguments.add_argument("epoch", help="number of epoches", type=int)
arguments.add_argument("--stage-2-path", help="path to stage 2 checkpoint", default=None)
arguments.add_argument("--roi_path", help="roi data", default="data/ROIs")
arguments.add_argument("--batch-size", help="batch size", type=int, default=2)
arguments.add_argument("--rois-per-batch", help="number of rois per batch", type=int, default=128)
arguments.add_argument("-e", "--experiments", help="experiment path", default="data/Experiments")
arguments.add_argument("-d", "--display", help="display interval", type=int, default=100)
arguments.add_argument("-r", "--resume", help="resume from a checkpoint", default=None)
arguments.add_argument("--lr", help="base learning rate", type=float, default=0.001)
arguments.add_argument("--momentum", help="momentum constant", type=float, default=0.9)
arguments.add_argument("--gamma", help="learning rate decay constant", type=float, default=0.1)
arguments.add_argument("--lr-stepsize", help="learning rate decay stepsize", type=int, default=11)
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

    assert args.rois_per_batch % args.batch_size == 0, "Uneven number of rois per image"
    rois_per_image = args.rois_per_batch / args.batch_size
    train_data = VOCDataSetROIs("data", "train", rois_per_image)
    dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=5, collate_fn=collate_rois_fn)

    ################### MODEL BOOTSRAP #####################
    print("[+] Bootstrapping model")
    if args.stage_2_path is not None:
        print("[+] Loading stage 2 weights")
    net = FasterRCNN(args.stage_2_path).cuda()
    net.train()

    if args.resume is not None:
        print("[+] Resuming from %s" % args.resume)
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint["state_dict"])

    cross_entropy = CrossEntropyLoss(size_average=True).cuda()
    smooth_l1_loss = SmoothL1Loss(size_average=False).cuda()
    optimizer = opt.SGD([params for params in net.parameters() if params.requires_grad],
                        lr=args.lr, momentum=args.momentum, weight_decay=0.0005)

    ################### MODEL TRAINING #####################
    print("[+] Training model")
    start_epoch = 0 if args.resume is None else checkpoint["epoch"]
    for epoch in xrange(start_epoch, args.epoch):
        adjust_learning_rate(optimizer, epoch)
        train(net, cross_entropy, smooth_l1_loss, optimizer, dataloader, experiment_env, epoch)


def train(net, cross_entropy, smooth_l1_loss, optimizer, dataloader, experiment, epoch):
    epoch_start_time = time.time() 
    epoch_loss = AverageMeter()
    epoch_cls_loss = AverageMeter()
    epoch_l1_loss = AverageMeter()

    for i, (_, images, rois, roi_labels, regression_targets, regression_weights, image_info) in enumerate(dataloader):
        optimizer.zero_grad()

        images = Variable(images.cuda())
        rois = Variable(rois.cuda())
        roi_labels = Variable(roi_labels.cuda())
        regression_targets = Variable(regression_targets.cuda())
        regression_weights = Variable(regression_weights.cuda())

        cls_scores, bbox_preds = net(images, rois)

        # multi-class classification loss
        cross_entropy_loss = cross_entropy(cls_scores, roi_labels)

        # bbox classification loss
        regression_targets = torch.mul(regression_targets, regression_weights)
        bbox_preds = torch.mul(bbox_preds, regression_weights) 
        bbox_loss = smooth_l1_loss(bbox_preds, regression_targets) / (regression_targets.size(0) + 1e-4)

        loss = cross_entropy_loss + bbox_loss
        loss.backward()
        optimizer.step()

        epoch_loss.update(loss.data[0])
        epoch_cls_loss.update(cross_entropy_loss.data[0])
        epoch_l1_loss.update(bbox_loss.data[0])
        if i % args.display == (args.display - 1):
            print("Epoch %d/%d (step %d): ave_loss=%.3f cls_loss=%.3f l1_loss=%.3f" %
                  (epoch + 1, args.epoch, i, epoch_loss.ave, epoch_cls_loss.ave, epoch_l1_loss.ave))

    total_time = epoch_start_time - time.time()
    print("Epoch %d/%d: ave_loss=%.3f total_time=%.3f" % (epoch + 1, args.epoch, epoch_loss.ave, total_time))
    torch.save({
        "epoch": epoch,
        "ave_loss": epoch_loss.ave,
        "state_dict": net.state_dict()
    }, join(experiment["checkpoint"], "checkpoint_%d.tar" % epoch))


if __name__ == "__main__":
    main()
