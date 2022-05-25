"""Collection of functions for calculating segmentation metrics

This includes:
- DSC coefficient
- Hausdorff distance

"""
import csv
import json
import os
import cv2
import json
import argparse
import numpy as np
import csv

from tools.variability_shared import *


def get_dsc_coef(capsule1_path, region1_path, capsule2_path, region2_path):
    """
    Compute DSC coefficient for given masks

    :param capsule1_path: path to capsule mask for first segmentation
    :param region1_path: path to region mask for first segmentation
    :param capsule2_path: path to capsule mask for second segmentation
    :param region2_path: path to region mask for second segmentation
    :return: list of DSC scores for each class
    """
    class_coefs = []
    capsules = []
    regions = []
    capsules.append(cv2.imread(capsule1_path))
    capsules.append(cv2.imread(capsule2_path))
    regions.append(cv2.imread(region1_path))
    regions.append(cv2.imread(region2_path))
    for i in range(1, 5):
        masks = generate_masks(capsules, regions, i, 2)
        dsc_coef = compute_dsc(masks[0], masks[1])
        class_coefs.append(dsc_coef)
    return class_coefs


def generate_score_csv(path1, path2, outpath, score_func=get_dsc_coef):
    """
    Compute DSC coefficients for mask pairs at given paths and save to
    CSV file

    :param path1: path first set of masks
    :param path2: path to second set of masks
    :param outpath: path to DSC score CSV file
    """
    with open(outpath, "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(["file", "Capsule", "Central Echo Complex",
                         "Medulla", "Cortex", "mean"])
        total = 0
        avg = np.array([0, 0, 0, 0])

        for filename in os.listdir(os.path.join(path1, "capsule")):
            capsule1_path = os.path.join(path1, "capsule", filename)
            region1_path = os.path.join(path1, "regions", filename)
            capsule2_path = os.path.join(path2, "capsule", filename)
            region2_path = os.path.join(path2, "regions", filename)
            coefs = score_func(capsule1_path, region1_path,
                                 capsule2_path, region2_path)
            mean = np.array(coefs).mean()
            row = [filename] + coefs + [mean]
            writer.writerow(row)
            avg = avg + np.array(coefs)
            total += 1

        avg = avg / total
        mean = avg.sum() / 4
        writer.writerow(["mean", avg[0], avg[1], avg[2], avg[3], mean])


def get_precision(capsule1_path, region1_path, capsule2_path, region2_path):
    """
    Compute precision for given masks

    :param capsule1_path: path to capsule mask for first segmentation
    :param region1_path: path to region mask for first segmentation
    :param capsule2_path: path to capsule mask for second segmentation
    :param region2_path: path to region mask for second segmentation
    :return: list of precisions for each class
    """
    class_coefs = []
    capsules = []
    regions = []
    capsules.append(cv2.imread(capsule1_path))
    capsules.append(cv2.imread(capsule2_path))
    regions.append(cv2.imread(region1_path))
    regions.append(cv2.imread(region2_path))
    for i in range(1, 5):
        masks = generate_masks(capsules, regions, i, 2)
        precision_coef = compute_precision(masks[0], masks[1])
        class_coefs.append(precision_coef)
    return class_coefs


def get_recall(capsule1_path, region1_path, capsule2_path, region2_path):
    """
    Compute recall for given masks

    :param capsule1_path: path to capsule mask for first segmentation
    :param region1_path: path to region mask for first segmentation
    :param capsule2_path: path to capsule mask for second segmentation
    :param region2_path: path to region mask for second segmentation
    :return: list of recall for each class
    """
    class_coefs = []
    capsules = []
    regions = []
    capsules.append(cv2.imread(capsule1_path))
    capsules.append(cv2.imread(capsule2_path))
    regions.append(cv2.imread(region1_path))
    regions.append(cv2.imread(region2_path))
    for i in range(1, 5):
        masks = generate_masks(capsules, regions, i, 2)
        recall_coef = compute_recall(masks[0], masks[1])
        class_coefs.append(recall_coef)
    return class_coefs


def get_hd(capsule1_path, region1_path,
           capsule2_path, region2_path, conversion):
    """
    Compute Hausdorff distance for given masks

    :param capsule1_path: path to capsule mask for first segmentation
    :param region1_path: path to region mask for first segmentation
    :param capsule2_path: path to capsule mask for second segmentation
    :param region2_path: path to region mask for second segmentation
    :param conversion: pixel to mm conversions
    :return: list of Hausdorff distances for each class
    """
    class_coefs = []
    capsules = []
    regions = []
    capsules.append(cv2.imread(capsule1_path))
    capsules.append(cv2.imread(capsule2_path))
    regions.append(cv2.imread(region1_path))
    regions.append(cv2.imread(region2_path))
    for i in range(1, 4+1):
        masks = generate_masks(capsules, regions, i, 2)
        hd = compute_hd(masks[0], masks[1])
        class_coefs.append(hd * conversion)
    return class_coefs


def generate_hd_csv(path1, path2, converter, outpath):
    """
    Compute Hausdorff distances for mask pairs at given paths and save to
    CSV file

    :param path1: path first set of masks
    :param path2: path to second set of masks
    :param converter: map of pixel to mm conversions
    :param outpath: path to Hausdorff distance CSV file
    """
    with open(outpath, "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(["file", "Capsule", "Central Echo Complex",
                         "Medulla", "Cortex", "mean"])
        total = 0
        avg = np.array([0, 0, 0, 0])

        for filename in os.listdir(os.path.join(path1, "capsule")):
            capsule1_path = os.path.join(path1, "capsule", filename)
            region1_path = os.path.join(path1, "regions", filename)
            capsule2_path = os.path.join(path2, "capsule", filename)
            region2_path = os.path.join(path2, "regions", filename)
            conversion = float(converter[filename])
            coefs = get_hd(capsule1_path, region1_path,
                           capsule2_path, region2_path, conversion)
            mean = np.array(coefs).mean()
            row = [filename] + coefs + [mean]
            writer.writerow(row)
            avg = avg + np.array(coefs)
            total += 1

        avg = avg / total
        mean = avg.sum()/4
        writer.writerow(["mean", avg[0], avg[1], avg[2], avg[3], mean])