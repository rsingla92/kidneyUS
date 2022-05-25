"""Collection of functions for calculating size stats from kidney masks

This includes:
- Kidney mask length
- Kidney sweep length
- Areas of different classes

"""
import csv
import json
import argparse
import os
import numpy as np
import cv2
import math
import radiomics
from scipy import spatial

from tools.variability_shared import *


def get_mask_area(mask, pixel_dim):
    return np.sum(mask) * np.product(pixel_dim)


def get_mask_diameter(mask, pixel_dim):
    """
    Get major, minor axis length and bounding box for given mask

    :param mask: capsule mask
    :param pixel_dim: pixel to mm conversion
    :return:  major, minor axis length, bounding box
    """
    if(mask.sum() < 5):
        return 0, 0, None

    mask = mask.astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(mask[:, :, 0], cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    biggest_contour = contours[0]
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    if contour_sizes:
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    rect = cv2.minAreaRect(biggest_contour)
    (x, y), (width, height), angle = rect
    box = cv2.boxPoints(rect)

    if width > height:
        major = width
        minor = height
    else:
        major = height
        minor = width
    return major * pixel_dim, minor * pixel_dim, box


def get_kidney_lengths_by_area(sweep_dir, pixel_dim):
    """
    Get transverse and logitudinal lengths from sweep

    :param sweep_dir: directory with capsule masks for sweep
    :param pixel_dim: pixel to mm conversion
    :return: max length, min length across sweep
    """
    longitudinal_length = 0
    transverse_length_1 = 0
    transverse_length_2 = 0
    longitudinal_frame = 0
    transverse_frame = 0
    longitudinal_box = None
    transverse_box = None
    max_area = 0
    for f in os.listdir(sweep_dir):
        mask = cv2.imread(os.path.join(sweep_dir, f))
        major, minor, box = get_mask_diameter(mask, pixel_dim)
        area = get_mask_area(mask, pixel_dim)
        if area > max_area:
            longitudinal_length = major
            transverse_length_1 = minor
            if "frame" in f:
                longitudinal_frame = int(f.split("_")[2].replace("frame", ""))
            else:
                longitudinal_frame = -1
            longitudinal_box = box
            transverse_box = box
            max_area = area
        # if major > longitudinal_length:
        #     transverse_length_1 = minor
        #     # transverse_length_2 = major
        #     if "frame" in f:
        #         transverse_frame = int(f.split("_")[2].replace("frame", ""))
        #     else:
        #         frame = -1
        #     transverse_box = box
    
    transverse = (transverse_length_1, longitudinal_frame, transverse_box)
    longitudinal = (longitudinal_length, longitudinal_frame, longitudinal_box)

    return longitudinal, transverse


def get_kidney_lengths(sweep_dir, pixel_dim):
    """
    Get transverse and logitudinal lengths from sweep

    :param sweep_dir: directory with capsule masks for sweep
    :param pixel_dim: pixel to mm conversion
    :return: max length, min length across sweep
    """
    longitudinal_length = 0
    transverse_length_1 = 0
    transverse_length_2 = 0
    longitudinal_frame = 0
    transverse_frame = 0
    longitudinal_box = None
    transverse_box = None
    for f in os.listdir(sweep_dir):
        mask = cv2.imread(os.path.join(sweep_dir, f))
        major, minor, box = get_mask_diameter(mask, pixel_dim)
        if major > longitudinal_length:
            longitudinal_length = major
            transverse_length_1 = minor
            if "frame" in f:
                longitudinal_frame = int(f.split("_")[2].replace("frame", ""))
            else:
                longitudinal_frame = -1
            longitudinal_box = box
            transverse_box = box
        # if major > longitudinal_length:
        #     transverse_length_1 = minor
        #     # transverse_length_2 = major
        #     if "frame" in f:
        #         transverse_frame = int(f.split("_")[2].replace("frame", ""))
        #     else:
        #         frame = -1
        #     transverse_box = box
    
    transverse = (transverse_length_1, longitudinal_frame, transverse_box)
    longitudinal = (longitudinal_length, longitudinal_frame, longitudinal_box)

    return longitudinal, transverse

def get_kidney_volume(l, w ,d):
    return math.pi / 6 * l * w * d

def add_distance_line(overlay_path, box):
    """
    Draws bounding box onto images of kidney

    :param overlay_path: directory to overlay image annotate
    :param box: box contour to draw
    """
    img = cv2.imread(overlay_path)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (255, 255, 255), thickness=1)
    cv2.imwrite(overlay_path, img)


def get_mask_size_stats(capsule, region, overlay_path, pixel_dim):
    """
    Get length and area size stats for mask

    :param capsule: capsule mask
    :param region: region mask
    :param overlay_path: path to overlay of masks
    :param: pixel_dim: pixel to mm conversion
    """
    stats = []
    for i in range(1, 5):
        mask = generate_masks([capsule], [region], i, 1)[0]
        if i==1:
            major, minor, box = get_mask_diameter(mask, pixel_dim)
            if major > 0:
                add_distance_line(overlay_path, box)
            stats.append(major)
            stats.append(minor)
        area = get_mask_area(mask, pixel_dim)
        stats.append(area)
    return stats


def create_mask_size_stats_csv(inpath, overlay_path, converter, outfile):
    """
    Create CSV containing area and length size statistics for masks in the
    directory given by inpath

    :param inpath: path to mask directory
    :param overlay_path: path to overlay of masks
    :param converter: pixel to mm conversion dictionary
    :param: outfile: csv file for saving size stats
    """
    with open(outfile, "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(["File", "Major Axis", "Minor Axis", "Capsule Area",
                         "Central Echo Complex Area", "Medulla Area",
                         "Cortex Area"])
        for filename in os.listdir(os.path.join(inpath, "capsule")):
            capsule_path = os.path.join(inpath, "capsule", filename)
            region_path = os.path.join(inpath, "regions", filename)
            conversion = float(converter[filename])
            capsule = cv2.imread(capsule_path)
            region = cv2.imread(region_path)
            size_stats = get_mask_size_stats(
                    capsule, region, os.path.join(overlay_path, filename),
                    np.array([conversion, conversion]))
            writer.writerow([filename] + size_stats)
