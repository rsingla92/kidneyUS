"""Script to compute DSC scores for interrater variability

"""
import csv
import json
import argparse
import os
import numpy as np
import cv2
import sys

sys.path.append('../')

from tools.metric_utils import get_dsc_coef
from tools.variability_shared import *

NUM_CLASSES = 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute interrater DSC scores")
    parser.add_argument("csv_file", help="Repeats CSV file")
    parser.add_argument("datapath1",
                        help="First directory of files to compare")
    parser.add_argument("datapath2",
                        help="Second directory of files to compare")
    parser.add_argument("output_csv", help="DSC score CSV")
    args = parser.parse_args()
    extra_files = []
    try:
        extra_files = get_repeats(args.csv_file)
    except Exception as e:
        print("oh no")

    with open(args.output_csv, "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(["file", "Capsule", "Central Echo Complex", "Medulla", "Cortex", "mean"])
        total = 0
        avg = np.array([0, 0, 0, 0])

        capsule_path1 = os.path.join(args.datapath1, "capsule")

        for filename in os.listdir(capsule_path1):
            if filename not in extra_files:
                print(filename)
                capsule1_path = os.path.join(args.datapath1, "capsule", filename)
                region1_path = os.path.join(args.datapath1, "regions", filename)
                capsule2_path = os.path.join(args.datapath2,"capsule", filename)
                region2_path = os.path.join(args.datapath2, "regions", filename)
                coefs = get_dsc_coef(capsule1_path, region1_path, capsule2_path, region2_path)
                mean = np.array(coefs).mean()
                row = [filename] + coefs + [mean]
                writer.writerow(row)
                avg = avg + np.array(coefs)
                total += 1

        avg = avg / total
        mean = avg.sum()/NUM_CLASSES
        writer.writerow(["mean", avg[0], avg[1], avg[2], avg[3], mean])
