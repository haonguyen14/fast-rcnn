import xml.etree.ElementTree as ET
import pandas as pd

from sets import Set
from os import listdir
from os.path import join


def parse_annotation(root, image_dir):

    filename = join(image_dir, root.find("filename").text)
    objects = root.findall("object")
    return [(filename, parse_object_node(obj)) for obj in objects]


def parse_object_node(object_node):

    obj_class = object_node.find("name").text
    box_coor = object_node.find("bndbox")
    return (obj_class, [
        box_coor.find("xmin").text,
        box_coor.find("ymin").text,
        box_coor.find("xmax").text,
        box_coor.find("ymax").text,
    ])


def parse_file(filename, image_dir):

    return parse_annotation(
        ET.parse(filename).getroot(),
        image_dir)


def flatten_annotation(annotation):

    filename = annotation[0]
    obj_class = annotation[1][0]
    xmin, ymin, xmax, ymax = annotation[1][1]
    return [filename, obj_class, xmin, ymin, xmax, ymax]


def encoding_classes(annotations):

    unique_classes = Set([a[1] for a in annotations])
    unique_class_mapping = {
        classname: i for i, classname in enumerate(unique_classes)}

    return [[k, unique_class_mapping[k]] for k in unique_class_mapping.keys()], [
        [a[0], unique_class_mapping[a[1]], a[2], a[3], a[4], a[5]]
        for a in annotations
    ]


if __name__ == "__main__":

    annotation_dir = "data/Annotations/"
    image_dir = "data/JPEGImages/"
    preprocess_dir = "data/Preprocess/"

    file_paths = [
        join(annotation_dir, filename)
        for filename in listdir(annotation_dir)
    ]

    print("[+] Parsing %d annotation files" % len(file_paths))
    annotations = [parse_file(path, image_dir) for path in file_paths]
    annotations = sum(annotations, [])
    annotations = [flatten_annotation(a) for a in annotations]
    class_encoding, annotations = encoding_classes(annotations)

    print("[+] Found %d objects" % len(annotations))

    print("[+] Creating CSV file for annotations")
    dataframe = pd.DataFrame(
        annotations,
        columns=[
            "filename",
            "obj_class",
            "xmin",
            "ymin",
            "xmax",
            "ymax"
        ]
    )
    dataframe.to_csv(
        join(preprocess_dir, "annotations.csv"),
        index=False
    )

    print("[+] Creating CSV file for class mapping")
    class_encoding_dataframe = pd.DataFrame(
        class_encoding, columns=["class_name", "class_id"])
    class_encoding_dataframe.to_csv(
        join(preprocess_dir, "class_encoding.csv"),
        index=False
    )

    print("[+] Finished!")
