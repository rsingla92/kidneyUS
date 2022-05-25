"""Compute point stats

Computes mean and standard deviation of the points marked by the annotator

"""
import csv
import json
import argparse
import numpy as np

LABEL_MAP = {"Capsule": 0,
             "Central Echo Complex": 1,
             "Medulla": 2,
             "Cortex": 3}


def get_point_nums(f):
    """
    Get number of points for each image in annotation file

    :param f: path to annotation file
    :return: map of image to the number of points per class
    """
    point_dict = {}
    with open(f, 'r') as csv_fp:
        csv_reader = csv.DictReader(csv_fp)
        for row in csv_reader:
            filename = row["filename"]
            region_dict = row["region_shape_attributes"]
            class_dict = row["region_attributes"]
            reg_map = json.loads(region_dict)
            class_map = json.loads(class_dict)
            if class_map:
                reg_class = class_map["Anatomy"]
            if filename not in point_dict:
                point_dict[filename] = {"Capsule": 0,
                                        "Central Echo Complex": 0,
                                        "Medulla": 0,
                                        "Cortex": 0}
            if len(reg_map) > 0:
                if "all_points_x" in reg_map:
                    num_points = len(reg_map["all_points_x"])
                else:
                    num_points = 4
            else:
                num_points = 0
            point_dict[filename][reg_class] = \
                point_dict[filename][reg_class] + num_points
    return point_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get point statistics from data in VGG annotation file")
    parser.add_argument("infile", help="Input annotation file")
    parser.add_argument("outfile", help="Point stats file")

    args = parser.parse_args()
    point_dict = get_point_nums(args.infile)

    i = 0
    all_points = np.zeros((len(point_dict), 4))
    for f in point_dict.keys():
        for reg in point_dict[f].keys():
            all_points[i, LABEL_MAP[reg]] = point_dict[f][reg]
        i += 1
    mean = all_points.mean(axis=0)
    std = all_points.std(axis=0)
    with open(args.outfile, "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(["Class", "Mean", "Standard Deviation"])

        i = 0
        for label in LABEL_MAP.keys():
            writer.writerow([label, mean[i], std[i]])
            i += 1
