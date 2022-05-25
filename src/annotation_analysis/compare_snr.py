import os
import sys
import argparse
import cv2
import numpy as np

sys.path.append('../')

from tools.basic_utils import write_rows_to_file
from tools.distribution_utils import compute_snr_for_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare SNR of two sets of masks")
    parser.add_argument("img_path", help="Path to images")
    parser.add_argument("mask_path", help="Path to ground truth masks")
    parser.add_argument("prediction_path", help="Path to predicted masks")
    parser.add_argument("outpath", help="Output file with stats")
    args = parser.parse_args()

    rows = [["Filename",
             "CEC Mask SNR", "CEC Prediction SNR", "CEC SNR Difference",
             "Medulla Mask SNR", "Medulla Prediction SNR", "Medulla SNR Difference",
             "Cortex Mask SNR", "Cortex Prediction SNR", "Cortex SNR Difference",
             "Mean SNR Difference"]]

    count = 0
    diff_totals = [0, 0, 0, 0]
    for f in os.listdir(args.prediction_path):
        count += 1
        img = cv2.imread(os.path.join(args.img_path, f))
        mask = cv2.imread(os.path.join(args.mask_path, f))
        prediction = cv2.imread(os.path.join(args.prediction_path, f))

        mask_params = compute_snr_for_mask(img, mask, 3)
        prediction_params = compute_snr_for_mask(img, prediction, 3)

        diff_total = 0

        row = [f]
        for i in range(3): 
            row.append(mask_params[i])
            row.append(prediction_params[i])
            row.append(abs(mask_params[i]-prediction_params[i]))
            diff_totals[i] = diff_totals[i] + abs(mask_params[i]-prediction_params[i])
            diff_total = diff_total + abs(mask_params[i]-prediction_params[i])
        row.append(diff_total / 3)
        diff_totals[3] = diff_totals[3] + diff_total / 3
        rows.append(row)
    row = ["mean", "", "", diff_totals[0] / count,
           "", "", diff_totals[1] / count,
           "", "", diff_totals[2] / count,
           diff_totals[3] / count]
    rows.append(row)
    write_rows_to_file(args.outpath, rows)
