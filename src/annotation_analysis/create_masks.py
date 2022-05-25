"""Script to create masks from VGG annotations in CSV format

"""
import os
import cv2
import json
import argparse
import numpy as np
import sys
import csv
import sys

sys.path.append('../')

from tools.basic_utils import maybe_mkdir

sys.path.append('../')

from tools.basic_utils import maybe_mkdir

LABEL_MAP = {"Capsule": 1,
             "Central Echo Complex": 2,
             "Medulla": 3,
             "Cortex": 4}


def create_masks(mask_dict, img_path, mask1_path, mask2_path):
    """
    Create and save masks from annotations

    :param mask_dict: dictionary of polygon annotations
    :param img_path: path to image file
    :param mask1_path: path for capsule mask
    :param mask2_path: path for region mask
    """
    for key in mask_dict:
        img = cv2.imread(os.path.join(img_path, key))
        (height, width, channel) = img.shape
        mask1 = np.zeros((height, width))
        mask2 = np.zeros((height, width))
        for region in range(1, 5):
            if region in mask_dict[key]:
                points = mask_dict[key][region]
                if(region == 1):
                    cv2.fillPoly(mask1, points, color=region)
                else:
                    cv2.fillPoly(mask2, points, color=region-1)
        cv2.imwrite(os.path.join(mask1_path, key), mask1)
        cv2.imwrite(os.path.join(mask2_path, key), mask2)


def process_file(annotation_file):
    """
    Extract region shape information from annotation file

    :param annotation_file: CSV file of annotations
    :return: dictionary of extracted information
    """
    filename = ""
    mask_dict = {}
    with open(annotation_file, 'r') as csv_fp:
        csv_reader = csv.DictReader(csv_fp)
        for row in csv_reader:
            filename = row["filename"]
            if filename not in mask_dict:
                mask_dict[filename] = {}
            shape = json.loads(row["region_shape_attributes"])
            region_dict = json.loads(row["region_attributes"])
            if region_dict != {}:
                region = LABEL_MAP[region_dict["Anatomy"]]
            else:
                region = 0
            if region not in mask_dict[filename]:
                mask_dict[filename][region] = []
            if shape != {}:
                if(shape["name"] == "rect"):
                    x_points = [shape["x"]]
                    y_points = [shape["y"]]
                else:
                    x_points = shape["all_points_x"]
                    y_points = shape["all_points_y"]
                all_points = []
                for i, x in enumerate(x_points):
                    all_points.append([x, y_points[i]])
            mask_dict[filename][region].append(np.array(all_points, 'int32'))
    return mask_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create masks from CSV annotations")
    parser.add_argument("annotation_file",
                        help="Path to CSV file of annotations")
    parser.add_argument("img_path", help="Path to images that were annotated")
    parser.add_argument("mask_path", help="Path to directory for masks")
    args = parser.parse_args()
    mask_dict = process_file(args.annotation_file)
    img1_path = os.path.join(args.mask_path, "capsule")
    img2_path = os.path.join(args.mask_path, "regions")
    
    maybe_mkdir(args.mask_path)
    maybe_mkdir(img1_path)
    maybe_mkdir(img2_path)
    create_masks(mask_dict, args.img_path, img1_path, img2_path)
