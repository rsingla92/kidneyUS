"""Collection of functions for creating visuals for the segmentations

This includes:
- Overlaying the given masks on the original images
- Creating labelled compound images

"""
import os
import cv2
import json
import argparse
import numpy as np 
import csv

colour_map = {0: np.array([255, 255, 255]),  # Background
              1: np.array([255, 0, 0]),  # Capsule
              2: np.array([0, 255, 0]),  # Central Echo Complex
              3: np.array([0, 0, 255]),  # Medulla
              4: np.array([0, 255, 255])}  # Cortex


def create_compound_image(files, titles, outfile):
    """
    Creates compound image from given files and saves to specified file

    :param files: list of files to compoud (must have same height and width)
    :param titles: list of titles to use to label images (same length as files)
    :param outfile: file to save new image to
    """
    images = []

    for i in range(1, len(files)):
        img = cv2.imread(files[i])
        if len(img.shape) < 3:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        position = int(img.shape[1]/2 - 75)
        scale = img.shape[1] / 750

        cv2.putText(
            img, titles[i], (position, 50), font, scale, (255, 255, 255), 2)
        images.append(img)

    final = np.concatenate(tuple(images), axis=1)
    cv2.imwrite(outfile, final)


def create_compound_images(paths, titles, savepath):
    """
    Creates compound images with image from each path for all files in last
    indexed path in paths list. Files to merge together from each folder must
    have the same names.

    :param paths: list of paths to directories containing images to use
    :param titles: list of titles to use to label images (same length as paths)
    :param savepath: directory to save new image in
    """
    for filename in os.listdir(paths[-1]):
        files = []
        for p in paths:
            files.append(os.path.join(p, filename))
        outfile = os.path.join(savepath, filename)
        create_compound_image(files, titles, outfile)


def create_overlay(img_file, cap_mask_file, reg_mask_file, overlay_file):
    """
    Overlays coloured masks for kidney classes on original image

    :param img_file: original image file
    :param cap_mask_file: capsule mask file
    :param reg_mask_file: region mask file
    :param overlay_file: directory to save new image in
    """
    if not os.path.exists(img_file):
        return
    background = cv2.imread(img_file)
    if os.path.exists(cap_mask_file):
        foreground1 = cv2.imread(cap_mask_file)
    else:
        foreground1 = np.zeros(background.shape).astype('int8')
    if os.path.exists(reg_mask_file):
        foreground2 = cv2.imread(reg_mask_file)
    else:
        foreground2 = np.zeros(background.shape).astype('int8')
    foreground = np.zeros(background.shape)
    foreground[np.where(foreground2 > 0)] = foreground2[np.where(foreground2 > 0)] + 1
    foreground[np.where(foreground == 0)] = foreground1[np.where(foreground == 0)]
    (height, width, channel) = background.shape
    for i in range(height):
        for j in range(width):
            colour = colour_map[foreground[i][j][0]]
            for k in range(3):
                foreground[i][j][k] = colour[k]
    background = background.astype(foreground.dtype)
    overlay = cv2.addWeighted(background, 0.7, foreground, 0.1, 0)
    overlay[np.where((foreground1+foreground2)==0)] = background[np.where((foreground1+foreground2)==0)]

    cv2.imwrite(overlay_file, overlay)


def create_all_overlays(img_path, mask_path, overlay_path):
    """
    Overlays coloured masks for kidney classes on original image for
    all masks files in the mask_path directory

    :param img_path: directory containing original image files
    :param cap_mask_file: directory containing masks
    :param overlay_path: directory to save new images in
    """
    mask1_path = os.path.join(mask_path, "capsule")
    mask2_path = os.path.join(mask_path, "regions")
    for f in os.listdir(mask1_path):
        img_file = os.path.join(img_path, f)
        mask1_file = os.path.join(mask1_path, f)
        mask2_file = os.path.join(mask2_path, f)
        overlay_file = os.path.join(overlay_path, f)
        create_overlay(img_file, mask1_file, mask2_file, overlay_file)