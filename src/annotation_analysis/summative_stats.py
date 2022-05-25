"""Compute summative stats

This script computes and creates visuals for a variety of summative stats for
the given annotation CSV

It will perform the following:
- Number of annotated images
- Classes per frame
- Frames per class
- Number of pixels
- Pixel coverage (Total and relative)

"""
import csv
import json
import argparse
import os
import numpy as np
import cv2
import shutil
import matplotlib.pyplot as plt
import sys

sys.path.append('../')

from tools.variability_shared import *

NUM_CLASSES = 4
CLASSES = ["Capsule", "Central Echogenic Complex", "Medulla", "Cortex"]


def get_stats(capsule_path, region_path):
    """
    Get stats for given capsule and region masks

    :param capsule_path: path to capsule mask
    :param region_path: path to region mask
    :return: stats for the given masks
    """
    cap = cv2.imread(capsule_path)
    reg = cv2.imread(region_path)
    size = cap.shape[0] * cap.shape[1]
    total = cap + reg
    annotated = cap.sum() + reg.sum() > 0
    num_classes = len(np.unique(total)) - 1
    capsule = cap == 1
    cem = (reg == 1)
    medulla = (reg == 2)
    cortex = (reg == 3)
    classes = np.array([capsule.sum() > 0,
                        cem.sum() > 0,
                        medulla.sum() > 0,
                        cortex.sum() > 0])
    pixels = np.array([capsule.sum(), cem.sum(), medulla.sum(), cortex.sum()])
    percent_pixels = np.array([capsule.sum()/size,
                               cem.sum()/size,
                               medulla.sum()/size,
                               cortex.sum()/size])
    capsule_size = capsule.sum()
    if capsule_size > 0:
        rpercent_pixels = np.array([cem.sum()/capsule_size,
                                    medulla.sum()/capsule_size,
                                    cortex.sum()/capsule_size])
    else:
        rpercent_pixels = np.zeros(3)
    return annotated, num_classes, classes, pixels, percent_pixels, rpercent_pixels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summative stats calculator")
    parser.add_argument("csv_file", help="CSV file of repeated images")
    parser.add_argument("mask_path", help="Path to masks")
    parser.add_argument("output_dir", help="Output directory for stats")
    args = parser.parse_args()

    extra_files = get_repeats(args.csv_file)

    capsule_dir = os.path.join(args.mask_path, "capsule")
    region_dir = os.path.join(args.mask_path, "regions")

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.mkdir(args.output_dir)

    output_csv = os.path.join(args.output_dir, "summary.csv")

    with open(output_csv, "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(["Value", "Median", "Mean", "Max", "Min"])
        ann_count = 0
        total = 0
        class_totals = np.array([0, 0, 0, 0])
        num_files = len(os.listdir(capsule_dir)) - len(extra_files)
        total_num_classes = np.zeros(num_files)
        total_classes = np.zeros((num_files, 4))
        total_pixels = np.zeros((num_files, 4))
        total_percent_pixels = np.zeros((num_files, 4))
        total_rpercent_pixels = np.zeros((num_files, 3))
        for filename in os.listdir(capsule_dir):
            if filename not in extra_files:
                capsule_path = os.path.join(capsule_dir, filename)
                region_path = os.path.join(region_dir, filename)
                (annotated, num_classes, classes, pixels, percent_pixels, rpercent_pixels) = \
                    get_stats(capsule_path, region_path)
                if annotated:
                    ann_count += 1
                total_num_classes[total] = num_classes
                total_classes[total] = classes
                total_pixels[total] = pixels
                total_percent_pixels[total] = percent_pixels
                total_rpercent_pixels[total] = rpercent_pixels
                total += 1
            else:
                print(filename)
        writer.writerow(["Annotated Images", ann_count])

        writer.writerow(["Classes per Frame", np.median(total_num_classes),
                         total_num_classes.mean(), total_num_classes.max(),
                         total_num_classes.min()])

        plt.hist(total_num_classes, bins=[0, 1, 2, 3, 4, 5])
        plt.title('Histogram of the Number of Classes per Frame')
        plt.xlabel('Classes per Frame')
        plt.ylabel('Number of Frames')
        plt.savefig(os.path.join(args.output_dir, "classes_per_frame.png"))
        plt.clf()

        averaged_classes = total_classes.sum(axis=0)
        plt.bar(CLASSES, averaged_classes)
        plt.xlabel('Class')
        plt.ylabel('Frames')
        plt.title('Frames per Class')
        plt.savefig(os.path.join(args.output_dir, "frames_per_class.png"))
        plt.clf()

        for i in range(4):
            writer.writerow(["# Pixels: " + CLASSES[i],
                             np.median(total_pixels[:, i]),
                             total_pixels[:, i].mean(),
                             total_pixels[:, i].max(),
                             total_pixels[:, i].min()])

        for i in range(4):
            writer.writerow(["Pixel Coverage: " + CLASSES[i],
                             np.median(total_percent_pixels[:, i]),
                             total_percent_pixels[:, i].mean(),
                             total_percent_pixels[:, i].max(),
                             total_percent_pixels[:, i].min()])
            plt.hist(total_percent_pixels[:, i], bins=40)
            plt.title('Histogram of the Pixel Coverage of the ' + CLASSES[i] + ' Class')
            plt.xlabel('Ratio of Pixels in Image that are Classified as ' + CLASSES[i])
            plt.ylabel('Number of Frames')
            plt.savefig(os.path.join(args.output_dir, CLASSES[i] + "_pixel_coverage.png"))
            plt.clf()

        for i in range(1, 4):
            writer.writerow(["Relative Pixel Coverage: " + CLASSES[i],
                             np.median(total_rpercent_pixels[:, i-1]),
                             total_rpercent_pixels[:, i-1].mean(),
                             total_rpercent_pixels[:, i-1].max(),
                             total_rpercent_pixels[:, i-1].min()])
            plt.hist(total_rpercent_pixels[:, i-1], bins=40)
            plt.title('Histogram of the Pixel Coverage of the ' + CLASSES[i] + ' Class Relative to Capsule')
            plt.xlabel('Ratio of Pixels in Image that are Classified as ' + CLASSES[i])
            plt.ylabel('Number of Frames')
            plt.savefig(os.path.join(args.output_dir, CLASSES[i] + "_relative_pixel_coverage.png"))
            plt.clf()
