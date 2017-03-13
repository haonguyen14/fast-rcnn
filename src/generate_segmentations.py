import selectivesearch as ss
import pandas as pd
import numpy as np

from PIL import Image
from sets import Set
from os import listdir
from os.path import join

def get_regions(image_arr):

    h = image_arr.shape[0]
    w = image_arr.shape[1]
    scale = max(w, h)

    image_lbl, regions = ss.selective_search(image_arr, scale=scale)

    regions = Set([r["rect"] for r in regions])
    regions = list(regions)
    return [[r[0], r[1], r[0]+r[2], r[1]+r[3]] for r in regions if r[2] != 0 and r[3] != 0]

def is_positive(annotations, region):
    P = 0.5

    for a in annotations:
        if intersection(a, region) > P:
            return True

    return False

def intersection(ground_truth, bbox):

    gt_xmin = ground_truth[0]
    gt_ymin = ground_truth[1]
    gt_xmax = ground_truth[2]
    gt_ymax = ground_truth[3]
    gt_area = (gt_xmax - gt_xmin + 1) * (gt_ymax - gt_ymin + 1) * 1.0

    bb_xmin = bbox[0]
    bb_ymin = bbox[1]
    bb_xmax = bbox[2]
    bb_ymax = bbox[3]
    bb_area = (bb_xmax - bb_xmin + 1) * (bb_ymax - gt_ymin + 1) * 1.0

    intersect_xmin = max(gt_xmin, bb_xmin)
    intersect_ymin = max(gt_ymin, bb_ymin)
    intersect_xmax = min(gt_xmax, bb_xmax)
    intersect_ymax = min(gt_ymax, bb_ymax)

    if intersect_xmin > intersect_xmax or intersect_ymin > intersect_ymax:
        return 0.0

    intersect_area = (intersect_xmax - intersect_xmin + 1) * (intersect_ymax - intersect_ymin + 1) * 1.0
    total_area = (gt_area + bb_area) - intersect_area

    return intersect_area / total_area

if __name__ == "__main__":

    image_dir = "data/JPEGImages/"
    preprocess_dir = "data/Preprocess/Regions"
    annotation_dir = "data/Preprocess/Annotations"

    filenames = listdir(image_dir)
    print("[+] Found %d images" % len(filenames))

    for i, filename in enumerate(filenames):
        path = join(image_dir, filename)

        annotations = pd.read_csv(join(annotation_dir, filename + ".csv")).as_matrix().tolist()

        image_array = np.asarray(Image.open(path))
        h = image_array.shape[0]
        w = image_array.shape[1]

        regions = get_regions(image_array)
        regions = [
            [(r[0]*1.0)/w, (r[1]*1.0)/h, (r[2]*1.0)/w, (r[3]*1.0)/h, 1 if is_positive(annotations, r) else 0]
            for r in regions
        ]

        dataframe = pd.DataFrame(
           regions,
           columns=[
               "xmin",
               "ymin",
               "width",
               "height",
               "is_positive"
           ]
        )
        dataframe.to_csv(
            join(preprocess_dir, filename + ".csv"),
            index=False
        )

        percentage = (1.0 * i / len(filenames)) * 100
        if percentage % 2 == 0:
            print("[+] Processed %.1f percent of files" % percentage)

    print("[+] Finished!")
