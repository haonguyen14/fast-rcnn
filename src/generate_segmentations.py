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
    return [[r[0]*1.0/w, r[1]*1.0/h, r[2], r[3]] for r in regions if r[2] != 0 and r[3] != 0]

def get_positive_region(regions, p):
    pass

def get_negative_regions(regions, p):
    pass

if __name__ == "__main__":

    image_dir = "data/JPEGImages/"
    preprocess_dir = "data/Preprocess/Regions"

    filenames = listdir(image_dir)
    print("[+] Found %d images" % len(filenames))

    for i, filename in enumerate(filenames):
        path = join(image_dir, filename)
        regions = get_regions(np.asarray(Image.open(path)))

        dataframe = pd.DataFrame(
           regions,
           columns=[
               "xmin",
               "ymin",
               "width",
               "height"
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
