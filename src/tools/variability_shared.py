import csv
import numpy as np
import SimpleITK as sitk


def get_conversions(conversion_file):
    """
    Build dictionary containing pixel to mm conversions

    :param conversion_file: CSV file containing pixel to mm conversions
    :return: dictionary of pixel to mm conversions
    """
    conversions = {}
    with open(conversion_file, 'r') as csv_fp:
        csv_reader = csv.DictReader(csv_fp)
        for row in csv_reader:
            filename = row["Filename"]
            factor = row["mm/pixel"]
            conversions[filename] = factor
    return conversions


def get_repeat_sets(csv_file):
    """
    Get sets of repeated files for intrarater variability

    :param csv_file: CSV file with sets of repeated files
    :return: list of lists of repeated filesets
    """
    files = []
    with open(csv_file, 'r') as csv_fp:
        csv_reader = csv.DictReader(csv_fp)
        for row in csv_reader:
            fileset = []
            for i in range(1, 4):
                fileset.append(row["file" + str(i)])
            files.append(fileset)
    return files


def get_repeats(csv_file):
    """
    Get list of repeated files

    :param csv_file: CSV file with sets of repeated files
    :return: list of repeated files to exclude
    """
    files = []
    with open(csv_file, 'r') as csv_fp:
        csv_reader = csv.DictReader(csv_fp)
        for row in csv_reader:
            files.append(row["file2"])
            files.append(row["file3"])
    return files


def generate_masks(capsules, regions, cls, num):
    """
    Get separated masks for each class from segmentation file

    :param capsules: capsule segmentation
    :param regions: regions segmentation
    :param cls: class to extract
    :param num: number of segmentations in list
    :return: masks for specified class
    """
    masks = []

    if cls == 1:
        for i in range(num):
            masks.append(capsules[i]==cls)
    else:
        for i in range(num):
            masks.append(regions[i]==cls-1)
    return masks


def compute_dsc(mask1, mask2):
    """
    Compute DSC score for given masks

    :param mask1: first mask
    :param mask2: second mask
    :return: DSC score
    """
    intersection = np.logical_and(mask1, mask2)
    if(mask1.sum() + mask2.sum() != 0):
        dsc_coef = 2 * intersection.sum() / (mask1.sum() + mask2.sum())
    else:
        dsc_coef = 1
    return dsc_coef


def compute_precision(pred, truth):
    """
    Compute precision for given masks

    :param mask1: first mask
    :param mask2: second mask
    :return: precision
    """
    tp = np.logical_and(truth, pred).sum()
    fp = np.logical_and(np.logical_not(truth), pred).sum()
    if(tp.sum() == 0):
        precision = 0.0
    elif(truth.sum() + pred.sum() != 0):
        precision = tp / (tp + fp)
    else:
        precision = 1
    return precision


def compute_recall(pred, truth):
    """
    Compute precision for given masks

    :param mask1: first mask
    :param mask2: second mask
    :return: recall
    """
    tp = np.logical_and(truth, pred).sum()
    fn = np.logical_and(truth, np.logical_not(pred)).sum()
    if(tp.sum() == 0):
        recall = 0.0
    elif(truth.sum() + pred.sum() != 0):
        recall = tp / (tp + fn)
    else:
        recall = 1
    return recall


def compute_hd(mask1, mask2):
    """
    Compute Hausdorff distance for given masks

    :param mask1: first mask
    :param mask2: second mask
    :return: Hausdorff distance
    """
    if(mask1.sum() > 0 and mask2.sum() > 0):
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        img1 = sitk.GetImageFromArray(mask1.astype(int))
        img2 = sitk.GetImageFromArray(mask2.astype(int))
        hausdorff_distance_filter.Execute(img1, img2)
        hd = hausdorff_distance_filter.GetHausdorffDistance()
    else:
        hd = 0
    return hd
