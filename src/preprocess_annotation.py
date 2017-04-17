import xml.etree.ElementTree as ET
import pandas as pd

from sets import Set
from os import listdir
from os.path import join


def parse_annotation(root):

    objects = root.findall("object")
    return [parse_object_node(obj) for obj in objects]


def parse_object_node(object_node):

    obj_class = object_node.find("name").text
    obj_difficult = int(object_node.find("difficult").text)
    box_coor = object_node.find("bndbox")

    return [
        float(box_coor.find("xmin").text)-1,
        float(box_coor.find("ymin").text)-1,
        float(box_coor.find("xmax").text)-1,
        float(box_coor.find("ymax").text)-1,
        obj_difficult,
        obj_class
    ]


def parse_file(filename, image_dir):

    root = ET.parse(filename).getroot()
    filename = root.find("filename").text
    objects = parse_annotation(root)

    return (filename, objects)


def flatten_annotation(annotation):

    obj_class = annotation[0]
    xmin, ymin, xmax, ymax = annotation[1]
    return [xmin, ymin, xmax, ymax, obj_class]


def encoding_classes(file_annotations):

    unique_classes = [Set([a[5] for a in annotations]) for _, annotations in file_annotations]
    unique_classes = reduce(lambda x, y: x.union(y), unique_classes, Set())

    unique_class_mapping = {
        classname: i for i, classname in enumerate(unique_classes)}

    return [(key, unique_class_mapping[key]) for key in unique_class_mapping.keys()], [
        (
            filename,
            [[a[0], a[1], a[2], a[3], a[4], unique_class_mapping[a[5]]] for a in annotations]
        )
        for filename, annotations in file_annotations]


if __name__ == "__main__":

    annotation_dir = "data/Annotations/"
    image_dir = "data/JPEGImages/"
    preprocess_dir = "data/Preprocess/Annotations"

    file_paths = [
        join(annotation_dir, filename)
        for filename in listdir(annotation_dir)
    ]

    print("[+] Parsing %d annotation files" % len(file_paths))
    file_annotations = [parse_file(path, image_dir) for path in file_paths]
    class_encoding, file_annotations = encoding_classes(file_annotations)

    print("[+] Creating CSV files for annotations")
    for i, (filename, annotations) in enumerate(file_annotations):
        dataframe = pd.DataFrame(
            annotations,
            columns=[
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "difficult",
                "obj_class"
            ]
        )
        dataframe.to_csv(
            join(preprocess_dir, filename + ".csv"),
            index=False
        )

        percentage = (1.0 * i / len(file_annotations)) * 100
        if percentage % 10 == 0:
            print("\t[+] Created %.0f percent files" % percentage)

    print("[+] Creating CSV file for class mapping")
    class_encoding_dataframe = pd.DataFrame(
        class_encoding, columns=["class_name", "class_id"])
    class_encoding_dataframe.to_csv(
        join(preprocess_dir, "class_encoding.csv"),
        index=False
    )

    print("[+] Finished!")
