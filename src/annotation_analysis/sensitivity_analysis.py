import cv2
import os
import argparse
import sys
import numpy as np

sys.path.append('../')

from tools.metric_utils import get_dsc_coef
from tools.basic_utils import write_rows_to_file, format_floats_for_csv
from tools.variability_shared import *


def get_percent_change(val1, val2):
    if val1 == 0:
        return 0
    else:
        return abs(100 * (val2-val1) / val1)

def get_mask_sensitivity(regions):
    coefs = []
    for i in range(2, 5):
        new_coefs = []
        masks = generate_masks(None, regions, i, 2)
        new_coefs.append(compute_dsc(masks[0], masks[1]))
        kernel = np.ones((3, 3), np.uint8)
        og_mask = masks[1].copy().astype("uint8")
        masks[1] = cv2.erode(og_mask, kernel, iterations=1)
        new_coefs.append(compute_dsc(masks[0], masks[1]))
        new_coefs.append(get_percent_change(new_coefs[0], new_coefs[1]))
        masks[1] = cv2.erode(og_mask, kernel, iterations=10)
        new_coefs.append(compute_dsc(masks[0], masks[1]))
        new_coefs.append(get_percent_change(new_coefs[0], new_coefs[3]))
        masks[1] = cv2.dilate(og_mask, kernel, iterations=1)
        new_coefs.append(compute_dsc(masks[0], masks[1]))
        new_coefs.append(get_percent_change(new_coefs[0], new_coefs[5]))
        masks[1] = cv2.dilate(og_mask, kernel, iterations=10)
        new_coefs.append(compute_dsc(masks[0], masks[1]))
        new_coefs.append(get_percent_change(new_coefs[0], new_coefs[7]))
        coefs = coefs + format_floats_for_csv(new_coefs)
    return coefs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute sensitivity scores")
    parser.add_argument("csv_file", help="Repeats CSV file")
    parser.add_argument("datapath1",
                        help="First directory of files to compare")
    parser.add_argument("datapath2",
                        help="Second directory of files to compare")
    parser.add_argument("output_csv", help="Sensitivity score CSV")
    args = parser.parse_args()
    extra_files = []
    try:
        extra_files = get_repeats(args.csv_file)
    except Exception as e:
        print("oh no")

    rows = [["file", "Central Echo Complex",
             "Central Echo Complex - Erode 1",
             "%Change - Central Echo Complex - Erode 1",
             "Central Echo Complex - Erode 10",
             "%Change - Central Echo Complex - Erode 10",
             "Central Echo Complex - Dilate 1",
             "%Change - Central Echo Complex - Dilate 1",
             "Central Echo Complex - Dilate 10",
             "%Change - Central Echo Complex - Dilate 10",
             "Medulla", "Medulla - Erode 1", "%Change - Medulla - Erode 1",
             "Medulla - Erode 10",  "%Change - Medulla - Erode 10",
             "Medulla - Dilate 1", "%Change - Medulla - Dilate 1", 
             "Medulla - Dilate 10", "%Change - Medulla - Dilate 10",
             "Cortex", "Cortex - Erode 1", "%Change - Cortex - Erode 1",
             "Cortex - Erode 10", "%Change - Cortex - Erode 10",
             "Cortex - Dilate 1", "%Change - Cortex - Dilate 1",
             "Cortex - Dilate 10", "%Change - Cortex - Dilate 10"]]

    region_path = os.path.join(args.datapath2, "regions")
    total = 0
    avg = np.zeros(27)
    print(len(os.listdir(region_path)))
    for filename in os.listdir(region_path):
        # if filename not in extra_files:
        region1_path = os.path.join(args.datapath1, "regions", filename)
        region2_path = os.path.join(args.datapath2, "regions", filename)
        print(region1_path, region2_path)
        regions1 = cv2.imread(region1_path)
        regions2 = cv2.imread(region2_path)
        row = get_mask_sensitivity([regions1, regions2])

        avg = avg + np.array(row)
        row = [filename] + row
        total += 1
        rows.append(row)

    avg = avg / total
    row = ["mean"] + avg.tolist()
    rows.append(row)
    write_rows_to_file(args.output_csv, rows)
