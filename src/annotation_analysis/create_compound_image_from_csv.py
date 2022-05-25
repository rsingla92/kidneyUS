"""Creates compoud image from repeated file CSV file. Used for analyzing
interrater variability.
"""
import os
import cv2
import json
import argparse
import numpy as np 
import matplotlib.pyplot as plt

from variability_shared import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGG csv to json converter")
    parser.add_argument("original")
    parser.add_argument("overlay")
    parser.add_argument("csv_file")
    parser.add_argument("savepath")
    args = parser.parse_args()

    files = get_repeat_sets(args.csv_file)

    for fileset in files:
        file1 = os.path.join(args.original, fileset[0])
        file2 = os.path.join(args.overlay, fileset[0])
        file3 = os.path.join(args.overlay, fileset[1])
        file4 = os.path.join(args.overlay, fileset[2])
        filename = fileset[0].split("_")[0] + "_" + \
            fileset[1].split("_")[0] + "_" + fileset[2]
        outfile = os.path.join(args.savepath, filename)
        img1 = cv2.imread(file1)
        if len(img1.shape) < 3:
            img1 = np.repeat(img1[:, :, np.newaxis], 3, axis=2)
        img2 = cv2.imread(file2)
        img3 = cv2.imread(file3)
        img4 = cv2.imread(file4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = int(img1.shape[1]/2 - 300)
        scale = int(img1.shape[1] / 1000)
        y_pos = scale * 40
        cv2.putText(img1, 'Original', (position, y_pos), font,
                    scale, (255, 255, 255), 2)
        cv2.putText(img2, fileset[0], (position, y_pos), font,
                    scale, (255, 255, 255), 2)
        cv2.putText(img3, fileset[1], (position, y_pos), font,
                    scale, (255, 255, 255), 2)
        cv2.putText(img4, fileset[2], (position, y_pos), font,
                    scale, (255, 255, 255), 2)
        final = np.concatenate((img1, img2, img3, img4), axis=1)
        cv2.imwrite(outfile, final)
