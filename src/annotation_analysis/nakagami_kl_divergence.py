import os
import sys
import argparse
import cv2
import numpy as np

sys.path.append('../')

from tools.basic_utils import write_rows_to_file
from tools.distribution_utils import compute_nak_for_mask, compute_nakagami_kl_divergence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Nakagami distributions of two sets of masks")
    parser.add_argument("img_path", help="Path to images")
    parser.add_argument("mask_path", help="Path to ground truth masks")
    parser.add_argument("prediction_path", help="Path to predicted masks")
    parser.add_argument("outpath", help="Output file with stats")
    args = parser.parse_args()

    rows = [["Filename",
             "CEC Divergence",
             "Medulla Divergence",
             "Cortex Divergence",
             "Mean Divergence"]]

    count = 0
    div_totals = [0, 0, 0, 0]
    for f in os.listdir(args.prediction_path):
        print(f)
        count += 1
        img = cv2.imread(os.path.join(args.img_path, f))
        mask = cv2.imread(os.path.join(args.mask_path, f))
        prediction = cv2.imread(os.path.join(args.prediction_path, f))

        mask_params = compute_nak_for_mask(img, mask, 3)
        prediction_params = compute_nak_for_mask(img, prediction, 3)

        div_total = 0.0

        row = [f]
        for i in range(3):
            div = compute_nakagami_kl_divergence(mask_params[i], prediction_params[i]) 
            print("div")
            print(div)
            row.append(div)
            div_totals[i] = div_totals[i] + div
            div_total = div_total + div
        row.append(div_total / 3)
        rows.append(row)
    row = ["mean", "", "", div_totals[0] / count,
           "", "", div_totals[1] / count,
           "", "", div_totals[2] / count,
           "", "", div_totals[3] / count]
    rows.append(row)
    write_rows_to_file(args.outpath, rows)
