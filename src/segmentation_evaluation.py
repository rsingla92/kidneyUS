"""Segmentation Postprocessing

This script performs basic postprocessing for a kidney ultrasound test set
after segmentation.

It will perform the following:
- Compute DSC score for all classes
- Compute Hausdorff distance for all class
- Overlay masks on original image for visualization
- Create compound images to compare the annotation to the new segmnetation

"""
import csv
import json
import argparse
import os
import numpy as np
import cv2

from tools.variability_shared import *
from tools.size_tools import *
from tools.image_utils import *
from tools.metric_utils import generate_hd_csv, generate_score_csv, get_dsc_coef, get_precision, get_recall
from tools.basic_utils import maybe_mkdir

DSC_FILE_NAME = "dsc_scores.csv"
PRECISION_FILE_NAME = "precision.csv"
RECALL_FILE_NAME = "recall.csv"
HD_FILE_NAME = "hausdorff_scores.csv"
OVERLAY_DIR_NAME = "overlays"
COMPOUND_DIR_NAME = "compound"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to original images.")
    parser.add_argument("annotation_mask_path",
                        help="Path to masks from annotations")
    parser.add_argument("annotation_overlay_path",
                        help="Path to  overlays of annotations")
    parser.add_argument("mask_path",
                        help="Path to masks of segmentations to evaluate")
    parser.add_argument("metadata_file",
                        help="CSV file containing pixel to mm conversions")
    parser.add_argument("output_path",
                        help="Path to directory where output should be saved")
    args = parser.parse_args()

    maybe_mkdir(args.output_path)

    converter = get_conversions(args.metadata_file)
    print("Calculating DSC scores")
    generate_score_csv(args.mask_path, args.annotation_mask_path,
                     os.path.join(args.output_path, DSC_FILE_NAME),
                     score_func=get_dsc_coef)

    print("Calculating precision")
    generate_score_csv(args.mask_path, args.annotation_mask_path,
                     os.path.join(args.output_path, PRECISION_FILE_NAME),
                     score_func=get_precision)

    print("Calculating recall")
    generate_score_csv(args.mask_path, args.annotation_mask_path,
                     os.path.join(args.output_path, RECALL_FILE_NAME),
                     score_func=get_recall)

    print("Calculating Hausdorff Distances")
    generate_hd_csv(args.mask_path, args.annotation_mask_path, converter,
                    os.path.join(args.output_path, HD_FILE_NAME))

    print("Creating overlays")
    overlay_path = os.path.join(args.output_path, OVERLAY_DIR_NAME)
    maybe_mkdir(overlay_path)
    create_all_overlays(args.image_path, args.mask_path, overlay_path)

    print("Creating compound images")
    compound_path = os.path.join(args.output_path, COMPOUND_DIR_NAME)
    maybe_mkdir(compound_path)
    paths = [args.image_path, args.annotation_overlay_path, overlay_path]
    titles = ["Image", "Annotation", "Prediction"]
    create_compound_images(paths, titles, compound_path)
