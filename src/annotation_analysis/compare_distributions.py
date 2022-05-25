import os
import sys
import argparse
import cv2
import numpy as np

sys.path.append('../')

from tools.basic_utils import write_rows_to_file
from tools.distribution_utils import compute_nak_for_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Nakagami distributions of two sets of masks")
    parser.add_argument("img_path", help="Path to images")
    parser.add_argument("mask_path", help="Path to ground truth masks")
    parser.add_argument("prediction_path", help="Path to predicted masks")
    parser.add_argument("outpath", help="Output file with stats")
    args = parser.parse_args()

    rows = [["Filename",
             "CEC Mask Shape", "CEC Prediction Shape", "CEC Shape Difference",
             "CEC Mask Scale", "CEC Prediction Scale", "CEC Shape Scale",
             "Medulla Mask Shape", "Medulla Prediction Shape", "Medulla Shape Difference",
             "Medulla Mask Scale", "Medulla Prediction Scale", "Medulla Shape Scale",
             "Cortex Mask Shape", "Cortex Prediction Shape", "Cortex Shape Difference",
             "Cortex Mask Scale", "Cortex Prediction Scale", "Cortex Shape Scale",
             "Mean Mask Shape", "Mean Prediction Shape", "Mean Shape Difference",
             "Mean Mask Scale", "Mean Prediction Scale", "Mean Shape Scale",]]

    count = 0
    diff_totals = [0, 0, 0, 0, 0, 0, 0, 0]
    for f in os.listdir(args.prediction_path):
        print(f)
        count += 1
        img = cv2.imread(os.path.join(args.img_path, f))
        mask = cv2.imread(os.path.join(args.mask_path, f))
        prediction = cv2.imread(os.path.join(args.prediction_path, f))

        mask_params = compute_nak_for_mask(img, mask, 3)
        prediction_params = compute_nak_for_mask(img, prediction, 3)

        mask_shape_total = 0.0
        mask_scale_total = 0.0
        prediction_shape_total = 0.0
        prediction_scale_total = 0.0

        row = [f]
        for i in range(3): 
            row.append(mask_params[i][0])
            row.append(prediction_params[i][0])
            row.append(abs(mask_params[i][0]-prediction_params[i][0]))
            diff_totals[2*i] = diff_totals[2*i] + abs(mask_params[i][0]-prediction_params[i][0])
            row.append(mask_params[i][1])
            row.append(prediction_params[i][1])
            row.append(abs(mask_params[i][1]-prediction_params[i][1]))
            diff_totals[2*i+1] = diff_totals[2*i+1] + abs(mask_params[i][1]-prediction_params[i][1])
            mask_shape_total += mask_shape_total + mask_params[i][0]
            mask_scale_total += mask_scale_total + mask_params[i][1]
            prediction_shape_total += prediction_shape_total + prediction_params[i][0]
            prediction_scale_total += prediction_scale_total + prediction_params[i][1]
        row.append(mask_shape_total / 3)
        row.append(prediction_shape_total / 3)
        row.append(abs(mask_shape_total - prediction_shape_total) / 3)
        diff_totals[6] = diff_totals[6] + abs(mask_shape_total - prediction_shape_total) / 3
        row.append(mask_scale_total / 3)
        row.append(prediction_scale_total / 3)
        row.append(abs(mask_scale_total - prediction_scale_total) / 3)
        diff_totals[7] = diff_totals[7] + abs(mask_scale_total - prediction_scale_total) / 3
        rows.append(row)
    row = ["mean", "", "", diff_totals[0] / count,
           "", "", diff_totals[1] / count,
           "", "", diff_totals[2] / count,
           "", "", diff_totals[3] / count,
           "", "", diff_totals[4] / count,
           "", "", diff_totals[5] / count,
           "", "", diff_totals[6] / count,
           "", "", diff_totals[7] / count]
    rows.append(row)
    write_rows_to_file(args.outpath, rows)
