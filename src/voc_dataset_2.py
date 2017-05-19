import os
import cv2
import numpy as np
import numpy.random as npr
import pandas as pd
import pickle

import torch
from torch.utils.data import Dataset
from anchor_util import *

# BGR format to match VGG16 pretrained model
PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
INCLUDE_DIFFICULT_GT = False

CLASSES = ('__background__',
             'aeroplane', 'bicycle', 'bird', 'boat',
             'bottle', 'bus', 'car', 'cat', 'chair',
             'cow', 'diningtable', 'dog', 'horse',
             'motorbike', 'person', 'pottedplant',
             'sheep', 'sofa', 'train', 'tvmonitor')
NUM_CLASSES = len(CLASSES)
CLASS_TO_INDEX = dict(zip(CLASSES, xrange(NUM_CLASSES)))
INDEX_TO_CLASS = dict(zip(xrange(NUM_CLASSES), CLASSES))

ROI_BATCH_SIZE = 128
FG_FRACTION = 0.25
FG_THRESHOLD = 0.5
BG_THRESHOLD_LOW = 0.1
BG_THRESHOLD_HIGH = 0.5


def create_roi_batch_data(batch_size, sample_bboxes, sample_labels, sample_regression_targets, sample_regression_weights):
    bbox_indices = []
    for i, bboxes in enumerate(sample_bboxes):
        bbox_indices += ([i] * len(bboxes))
    bbox_indices = np.array(bbox_indices, dtype=float).reshape(-1, 1)
    
    bboxes = np.hstack((bbox_indices, np.vstack(sample_bboxes))) 
    labels = np.hstack(sample_labels)
    regression_targets = np.vstack(sample_regression_targets)
    regression_weights = np.vstack(sample_regression_weights)
    return bboxes, labels, regression_targets, regression_weights


def create_bbox_batch_data(batch_size, sample_bboxes, sample_labels):
    #  generating batch index for each bbox
    bbox_indices = []
    for i, bboxes in enumerate(sample_bboxes):
        bbox_indices += ([i] * len(bboxes))
    bbox_indices = np.array(bbox_indices, dtype=float).reshape(-1, 1)
    return np.hstack((bbox_indices, np.vstack(sample_bboxes))), np.hstack(sample_labels)


def create_image_batch_data(batch_size, sample_images):
    size = np.array([image.shape[1:] for image in sample_images]).max(axis=0)
    batch = np.zeros((batch_size, 3, size[0], size[1]), dtype=np.float32)
    for i, image in enumerate(sample_images):
        batch[i, :, 0:image.shape[1], 0:image.shape[2]] = image
    return batch


def collate_rois_fn(samples):
    batch_size = len(samples)
    indices = np.array([s[0] for s in samples], dtype=np.int)
    images = create_image_batch_data(batch_size, [s[1] for s in samples])
    bboxes, labels, regression_targets, regression_weights  = create_roi_batch_data(
            batch_size, [s[2] for s in samples], [s[3] for s in samples], [s[4] for s in samples], [s[5] for s in samples])
    image_info = np.vstack([s[6] for s in samples])

    return indices, \
           torch.Tensor(images), \
           torch.Tensor(bboxes), \
           torch.LongTensor(labels.astype(np.long)), \
           torch.Tensor(regression_targets), \
           torch.Tensor(regression_weights), \
           torch.Tensor(image_info)


def collate_fn(samples):
    batch_size = len(samples)
    indices = np.array([s[0] for s in samples], dtype=np.int)
    images = create_image_batch_data(batch_size, [s[1] for s in samples])
    gt_bboxes, gt_labels = create_bbox_batch_data(batch_size, [s[2] for s in samples], [s[3] for s in samples])
    image_info = np.vstack([s[4] for s in samples])

    return indices, \
           torch.Tensor(images), \
           torch.Tensor(gt_bboxes), \
           torch.Tensor(gt_labels), \
           torch.Tensor(image_info)


class VOCDataSet(Dataset):

    def __init__(self, root, image_set, min_size=600, max_size=1000, enabled_flip=True):
        assert (image_set == "train") | (image_set == "val"), "Invalid image set"

        self._root = root
        self._min_size = min_size
        self._max_size = max_size
        self._enabled_flip = enabled_flip
        self._image_set = "trainval" if image_set == "train" else "test"
        self._image_dir = os.path.join(self._root, "JPEGImages")
        self._annotation_dir = os.path.join(self._root, "Preprocess/Annotations")

        with open(os.path.join(self._root, "ImageSets/Main/%s.txt" % self._image_set), "r") as f:
            self._dataset_index = {i: line[0:-1] for i, line in enumerate(f)}

    def _get_image(self, index):
        image_path = os.path.join(self._image_dir, "%s.jpg" % self._dataset_index[index])
        image = cv2.imread(image_path).astype(np.float32)
        image = image - PIXEL_MEANS

        #  resize image
        shorter_dim = float(np.min(image.shape[0:2]))
        longer_dim = float(np.max(image.shape[0:2]))
        scale_const = float(self._min_size) / shorter_dim
        if (scale_const * longer_dim) > self._max_size:
            scale_const = float(self._max_size) / longer_dim
        image = cv2.resize(image, None, None,
                        fx=scale_const, fy=scale_const,
                        interpolation=cv2.INTER_LINEAR)

        #  transpose to match pytorch convolution input
        image = image.transpose(2, 0, 1)
        return image, np.hstack((image.shape[1:], scale_const)) 

    def _get_annotation(self, index, image_info):
        annotation_path = os.path.join(self._annotation_dir, "%s.jpg.csv" % self._dataset_index[index])
        annotations = pd.read_csv(annotation_path)
        annotations = annotations.as_matrix()
        gt_idx = np.arange(0, annotations.shape[0])
        if not INCLUDE_DIFFICULT_GT:
            gt_idx = np.where(annotations[:, 4] == 0)[0]
        return annotations[gt_idx, 0:4]*image_info[2], annotations[gt_idx, 5]

    def get_image_name_from_index(self, index):
        index_, is_flipped = self._get_index(index)
        return self._dataset_index[index_] + ("_flipped" if is_flipped else "")

    def __getitem__(self, i):
        index, is_flipped = self._get_index(i)
        image, image_info = self._get_image(index)
        gt_bbox, gt_label = self._get_annotation(index, image_info)
        gt_bbox = gt_bbox.astype(np.float32)
        gt_label = np.array([CLASS_TO_INDEX[x] for x in gt_label])
        
        if is_flipped:
            image = image[:, :, ::-1]
            gt_bbox_x1 = gt_bbox[:, 0].copy()
            gt_bbox_x2 = gt_bbox[:, 2].copy()
            gt_bbox[:, 0] = image.shape[2] - gt_bbox_x2 - 1
            gt_bbox[:, 2] = image.shape[2] - gt_bbox_x1 - 1
            
        return i, image, gt_bbox, gt_label, image_info

    def __len__(self):
        if self._enabled_flip:
            return len(self._dataset_index) * 2  # take into account flipped images
        return len(self._dataset_index)
    
    def _get_index(self, i):
        if i < 0 or i >= self.__len__():
            raise IndexError("list index (%d) out of range" % i)
        if self._enabled_flip:
            orig_size = len(self._dataset_index)
            return i % orig_size, i >= orig_size  # (index, is_flipped)
        return i, False


class VOCDataSetROIs(Dataset):

    def __init__(self, root, image_set, rois_per_image, roi_path="data/ROIs", min_size=600, max_size=1000, enabled_flip=True):
        self._image_data = VOCDataSet(root, image_set, min_size, max_size, enabled_flip=enabled_flip)
        self._roi_path = roi_path
        self._rois_per_image = rois_per_image

    def _get_rois(self, index, image_info):
        image_name = self._image_data.get_image_name_from_index(index)
        with open(os.path.join(self._roi_path, "%s.pkl" % image_name), "r") as f:
            rois = pickle.load(f)
        return rois * image_info[2]

    def _get_overlap_classes(self, rois, gt_bbox, gt_label):
        overlaps = get_overlap(rois, gt_bbox)
        max_overlaps = overlaps.max(axis=1)
        max_overlaps_idx = overlaps.argmax(axis=1)

        overlap_classes = np.zeros((len(rois), NUM_CLASSES), dtype=np.float32)
        roi_idx = np.where(max_overlaps > 0)[0]
        overlap_classes[roi_idx, gt_label[max_overlaps_idx[roi_idx]]] = max_overlaps[roi_idx]
        return overlap_classes, gt_label[max_overlaps_idx], gt_bbox[max_overlaps_idx]

    def _get_bbox_regression_targets(self, targets, labels):
        idx = np.where(labels > 0)[0] 
        regression_targets = np.zeros((len(targets), 4*NUM_CLASSES), dtype=np.float32)
        regression_weights = np.zeros(regression_targets.shape, dtype=np.float32)
        for i in idx:
            cls = labels[i]
            start = 4 * cls
            end = start + 4
            regression_targets[i, start:end] = targets[i, :]
            regression_weights[i, start:end] = np.array([1., 1., 1., 1.])
        return regression_targets, regression_weights

    def _sample_rois(self, rois, gt_bbox, gt_label, image_info):
        overlap_classes, target_cls, target_gt = self._get_overlap_classes(rois, gt_bbox, gt_label)
        assert len(overlap_classes) == len(target_cls) & len(overlap_classes) == len(target_gt)

        fg_per_image = np.round(FG_FRACTION * self._rois_per_image)
        fg_overlap_idx = np.where(overlap_classes >= FG_THRESHOLD)[0]
        fg_this_image = int(np.minimum(len(fg_overlap_idx), fg_per_image))
        if len(fg_overlap_idx) > 0:
            fg_overlap_idx = npr.choice(fg_overlap_idx, size=fg_this_image, replace=False)

        bg_overlap_idx = np.where((overlap_classes >= BG_THRESHOLD_LOW) & (overlap_classes < BG_THRESHOLD_HIGH))[0]
        bg_this_image = self._rois_per_image - fg_this_image
        bg_this_image = int(np.minimum(len(bg_overlap_idx), bg_this_image))
        if len(bg_overlap_idx) > 0:
            bg_overlap_idx = npr.choice(bg_overlap_idx, size=bg_this_image, replace=False)

        indices = np.append(fg_overlap_idx, bg_overlap_idx)
        keep_rois = rois[indices]
        keep_labels = target_cls[indices]
        keep_labels[fg_this_image:] = 0  # background class

        bbox_regression_targets = compute_bbox_target(keep_rois, target_gt[indices])
        bbox_regression_targets, bbox_regression_weights = self._get_bbox_regression_targets(bbox_regression_targets, keep_labels)

        return keep_rois, keep_labels, bbox_regression_targets, bbox_regression_weights

    def __getitem__(self, index):
        _, image, gt_bbox, gt_label, image_info = self._image_data[index]
        rois = self._get_rois(index, image_info)
        rois, roi_labels, roi_bbox_targets, roi_bbox_weights = self._sample_rois(rois, gt_bbox, gt_label, image_info)
        return index, image, rois, roi_labels, roi_bbox_targets, roi_bbox_weights, image_info

    def __len__(self):
        return len(self._image_data)
