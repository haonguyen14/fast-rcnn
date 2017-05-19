import time
import pickle
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader
from faster_rcnn import FasterRCNN
from region_proposal import RegionProposalNetwork
from voc_dataset_2 import CLASSES, CLASS_TO_INDEX, VOCDataSet, collate_fn
from generate_proposal import generate
from proposal import ProposalGenerator
from anchor_util import generate_proposals, clip_boxes
from nms import py_cpu_nms

arguments = argparse.ArgumentParser("Evaluate Detections")
arguments.add_argument("name", help="evaluation name")
arguments.add_argument("fastrcnn", help="path to fast rcnn model")
arguments.add_argument("rpn", help="path to rpn model")
arguments.add_argument("-d", "--dataset", choices=["train", "val"], default="train")


def get_detections(rpn, net, dataset, dataloader, nms_threshold=0.3, display=500):
    proposal_generator = ProposalGenerator()
    softmax_fast_rcnn = nn.Softmax().cuda()
    softmax_rpn = nn.Softmax2d().cuda()

    start_time = time.time()
    detection_results = {classname: {} for classname in CLASSES if classname != "__background__"}
    for i, (indices, image_arr, _, _, image_info) in enumerate(dataloader):
        image_name = dataset.get_image_name_from_index(indices[0])
        bboxes = generate(rpn, softmax_rpn, proposal_generator, image_arr, image_info[0, :])
        bboxes = bboxes * image_info[0, -1]  # scale bboxes
        proposals = np.hstack((np.zeros(len(bboxes)).reshape(-1, 1), bboxes))
        proposals = autograd.Variable(torch.Tensor(proposals).cuda())
        image_arr = autograd.Variable(image_arr.cuda())
        cls_scores, bbox_deltas = net(image_arr, proposals)
        cls_preds = softmax_fast_rcnn(cls_scores)
        cls_preds = cls_preds.cpu().data.numpy()
        bbox_preds = generate_proposals(bboxes, bbox_deltas.cpu().data.numpy())
        bbox_preds = clip_boxes(bbox_preds, image_info[0, 0], image_info[0, 1])
        
        for classname in detection_results:
            class_idx = CLASS_TO_INDEX[classname]
            idx = np.where(cls_preds[:, class_idx] > 0.05)[0]
            scores = cls_preds[idx, class_idx]
            bboxes = bbox_preds[idx, (class_idx*4):(class_idx*4+4)]
            detections = np.hstack((bboxes, scores.reshape(-1, 1))).astype(np.float32, copy=False)
            keep_idx = py_cpu_nms(detections, nms_threshold)
            detections = detections[keep_idx, :]
            detection_results[classname][image_name] = detections

        if i % display == (display - 1):
            total_time = time.time() - start_time
            ave_time = total_time / display
            print("[+] Processed %d. Average time=%.3f/image" % (i, ave_time))
            start_time = time.time()

    return detection_results


def count_tp_fp(gt_bboxes, detections, threshold=0.5):
    det_scores = detections[:, -1]
    det_bboxes = detections[:, :4]

    # sort in decreasing order
    ids = np.argsort(-det_scores)
    det_scores = det_scores[ids]
    det_bboxes = det_bboxes[ids, :]

    is_detected = [False] * len(gt_bboxes)
    assignments = [None] * len(ids)
    for i in range(len(ids)):
        max_overlap = -np.inf
        if len(gt_bboxes) > 0:
            xmin = np.maximum(gt_bboxes[:, 0], det_bboxes[i, 0])
            ymin = np.maximum(gt_bboxes[:, 1], det_bboxes[i, 1])
            xmax = np.minimum(gt_bboxes[:, 2], det_bboxes[i, 2])
            ymax = np.minimum(gt_bboxes[:, 3], det_bboxes[i, 3])
            w = np.maximum(xmax - xmin + 1., 0.)
            h = np.maximum(ymax - ymin + 1., 0.)

            intersection = w * h
            union = ((det_bboxes[i, 2] - det_bboxes[i, 0] + 1.) * (det_bboxes[i, 3] - det_bboxes[i, 1] + 1.)) + \
                    ((gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1.) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1.)) - intersection

            overlaps = intersection / union
            max_overlap = np.max(overlaps)
            max_overlap_arg = np.argmax(overlaps)

        if max_overlap > threshold:
            if not is_detected[max_overlap_arg]:
                assignments[i] = (det_scores[i], 1, 0)  # (score, is_true_pos, is_false_pos)
                is_detected[max_overlap_arg] = True
            else:
                assignments[i] = (det_scores[i], 0, 1)  # (score, is_true_pos, is_false_pos)
        else:
            assignments[i] = (det_scores[i], 0, 1)  # (score, is_true_pos, is_false_pos)

    return assignments


def compute_recall_precision(assignments, npos):
    scores = np.array([x[0] for x in assignments], dtype=np.float32)
    tp = np.array([x[1] for x in assignments], dtype=np.float32)
    fp = np.array([x[2] for x in assignments], dtype=np.float32)

    # merge all detection assignments in decreasing order
    ids = np.argsort(-scores)
    tp = tp[ids]
    fp = fp[ids]

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / float(npos)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    return recalls, precisions


def compute_ap(recalls, precisions):
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0.
        else:
            p = np.max(precisions[recalls >= t])
        ap += (p / 11.)
    return ap


def get_ap(rpn, net, dataset, dataloader, name, nms_threshold=0.3, tempdir="data/Evaluations"):
    detection_path = os.path.join(tempdir, "detections_%s.pkl" % name)
    if os.path.exists(detection_path):
        with open(detection_path, "r") as f:
            detections = pickle.load(f)
    else:
        detections = get_detections(rpn, net, dataset, dataloader, nms_threshold, display=500)
        with open(detection_path, "wb") as f:
            pickle.dump(detections, f)

    ave_ap = 0.
    for classname in detections:
        npos = 0
        assignments = []
        for indices, image_arr, gt_bboxes, gt_labels, image_info in dataloader:
            image_name = dataset.get_image_name_from_index(indices[0])
            gt_bboxes = gt_bboxes.numpy()
            gt_labels = gt_labels.numpy()
            gt_idx = np.where(gt_labels == CLASS_TO_INDEX[classname])[0]
            gt_bboxes = gt_bboxes[gt_idx, 1:]
            assignments_ = count_tp_fp(gt_bboxes, detections[classname][image_name])
            assignments += assignments_
            npos += len(gt_bboxes)
        recalls, precisions = compute_recall_precision(assignments, npos)
        ap = compute_ap(recalls, precisions)
        print("%s: AP=%.3f" % (classname, ap))
        ave_ap += ap
    return ave_ap / float(len(CLASSES))


def main():
    args = arguments.parse_args()

    dataset = VOCDataSet("data", args.dataset, enabled_flip=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=5, collate_fn=collate_fn)

    print("[+] Bootstrapping model")
    print("[+] Loading rpn model from %s" % args.rpn)
    print("[+] Loading fast rcnn model from %s" % args.fastrcnn)
    fast_rcnn_checkpoint = torch.load(args.fastrcnn)
    rpn_checkpoint = torch.load(args.rpn)
    fast_rcnn = FasterRCNN().cuda()
    fast_rcnn.load_state_dict(fast_rcnn_checkpoint["state_dict"])
    fast_rcnn.eval()
    rpn = RegionProposalNetwork().cuda()
    rpn.load_state_dict(rpn_checkpoint["state_dict"])

    print("[+] Calculating Average Precision")
    ap = get_ap(rpn, fast_rcnn, dataset, dataloader, name="%s_%s" % (args.dataset, args.name))
    print("Average Precision=%.3f" % ap)


if __name__ == "__main__":
    main()