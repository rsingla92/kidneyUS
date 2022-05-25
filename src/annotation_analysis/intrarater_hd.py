"""Script to compute Hausdorff distance for intrarater variability

"""
import csv
import json
import argparse
import os
import numpy as np
import cv2
import sys

sys.path.append('../')

from tools.variability_shared import *

NUM_CLASSES = 4
NUM_FILES = 3


def get_hd(pair, capsule_path, region_path, conversion):
    """
    Compute Hausdorff distance for given masks

    :param pair: set of images to compare
    :param capsule_path: path to region mask for first segmentation
    :param region_path: path to capsule mask for second segmentation
    :param conversion: pixel to mm conversions
    :return: list of Hausdorff distances for each class
    """
    class_coefs = []
    capsules = []
    regions = []
    for i in range(3):
        capsules.append(cv2.imread(os.path.join(capsule_path, pair[i])))
        regions.append(cv2.imread(os.path.join(region_path, pair[i])))

    for i in range(1, NUM_CLASSES+1):
        masks = generate_masks(capsules, regions, i, 3)
        hd_total = 0
        for pair in [(0, 1), (1, 2), (0, 2)]:
            mask1 = masks[pair[0]]
            mask2 = masks[pair[1]]
            hd = compute_hd(mask1, mask2)
            hd_total += hd
        class_coefs.append(hd_total/NUM_FILES * conversion)
    return class_coefs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute intrarater Hausdorff distances")
    parser.add_argument("csv_file", help="Path to CSV with repeated files")
    parser.add_argument("conversion_file", help="Path to CSV with repeated files")
    parser.add_argument("mask_path", help="Path to mask files")
    parser.add_argument("output_csv", help="Path to Hausdorff distance CSV")
    args = parser.parse_args()

    capsule_path = os.path.join(args.mask_path, "capsule")
    region_path = os.path.join(args.mask_path, "regions")

    files = get_repeat_sets(args.csv_file)
    converter = get_conversions(args.conversion_file)
    with open(args.output_csv, "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(["file1", "file2", "file3", "Capsule",
                         "Central Echo Complex", "Medulla",
                         "Cortex", "mean"])
        total = len(files)
        avg = np.array([0, 0, 0, 0])
        for group in files:
            conversion = float(converter[group[0]])
            coefs = get_hd(group, capsule_path, region_path, conversion)
            mean = np.array(coefs).mean()
            row = group + coefs + [mean]
            writer.writerow(row)
            avg = avg + np.array(coefs)
        avg = avg / total
        mean = avg.sum()/NUM_CLASSES
        writer.writerow(["mean", "mean", "mean", avg[0], avg[1],
                         avg[2], avg[3], mean])
