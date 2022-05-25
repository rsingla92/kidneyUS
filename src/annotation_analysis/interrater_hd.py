"""Script to compute Hausdorff distance for interrater variability

"""
import csv
import json
import argparse
import os
import numpy as np
import cv2
import sys

sys.path.append('../')

from tools.metric_utils import get_hd
from tools.variability_shared import *

NUM_CLASSES = 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute interrater Hausdorff distances")
    parser.add_argument("csv_file", help="CSV of repeated images")
    parser.add_argument("conversion_file",
                        help="CSV file containing pixel to mm conversions")
    parser.add_argument("path1", help="First directory of files to compare")
    parser.add_argument("path2", help="Second directory of files to compare")
    parser.add_argument("output_csv", help="Hausdorff distance CSV")
    args = parser.parse_args()

    extra_files = get_repeats(args.csv_file)
    converter = get_conversions(args.conversion_file)

    with open(args.output_csv, "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(["file", "Capsule", "Central Echo Complex", "Medulla", "Cortex", "mean"])
        total = 0
        avg = np.array([0, 0, 0, 0])

        for filename in os.listdir(os.path.join(args.path1, "capsule")):
            if filename not in extra_files:
                capsule1_path = os.path.join(args.path1, "capsule", filename)
                region1_path = os.path.join(args.path1, "regions", filename)
                capsule2_path = os.path.join(args.path2, "capsule", filename)
                region2_path = os.path.join(args.path2, "regions", filename)
                conversion = float(converter[filename])
                coefs = get_hd(capsule1_path, region1_path, capsule2_path, region2_path, conversion)
                mean = np.array(coefs).mean()
                row = [filename] + coefs + [mean]
                writer.writerow(row)
                avg = avg + np.array(coefs)
                total += 1

        avg = avg / total
        mean = avg.sum()/NUM_CLASSES
        writer.writerow(["mean", avg[0], avg[1], avg[2], avg[3], mean])
