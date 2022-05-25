import os
import argparse
import cv2
import numpy as np

from tools.basic_utils import maybe_mkdir


def restrict_segmentations(mask_path, output_path):
    cap_img_path = os.path.join(mask_path, "capsule")
    reg_img_path = os.path.join(mask_path, "regions")
    maybe_mkdir(os.path.join(output_path, "capsule"))
    maybe_mkdir(os.path.join(output_path, "regions"))
    for f in os.listdir(cap_img_path):
        cap_img = cv2.imread(os.path.join(cap_img_path, f))
        reg_img = cv2.imread(os.path.join(reg_img_path, f))
        cap_img = cv2.cvtColor(cap_img, cv2.COLOR_BGR2GRAY)
        reg_img = cv2.cvtColor(reg_img, cv2.COLOR_BGR2GRAY)
        reg_img = reg_img * cap_img
        cv2.imwrite(os.path.join(output_path, "capsule", f), cap_img)
        cv2.imwrite(os.path.join(output_path, "regions", f), reg_img)

def clean_segmentations(mask_path, output_path):
    cap_img_path = os.path.join(mask_path, "capsule")
    reg_img_path = os.path.join(mask_path, "regions")
    maybe_mkdir(os.path.join(output_path, "capsule"))
    maybe_mkdir(os.path.join(output_path, "regions"))
    for f in os.listdir(cap_img_path):
        cap_img = cv2.imread(os.path.join(cap_img_path, f))
        reg_img = cv2.imread(os.path.join(reg_img_path, f))
        cap_img = cv2.cvtColor(cap_img, cv2.COLOR_BGR2GRAY)
        # reg_img = cv2.cvtColor(reg_img, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(
            cap_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        new_cap_img = np.zeros(cap_img.shape)
        if contour_sizes:
            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
            cv2.fillPoly(new_cap_img, pts=[biggest_contour], color=1)
        reg_img = reg_img
        cv2.imwrite(os.path.join(output_path, "capsule", f), new_cap_img)
        cv2.imwrite(os.path.join(output_path, "regions", f), reg_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mask_path", help="Path to original images.")
    parser.add_argument("output_path",
                        help="Path to masks from annotations")
    args = parser.parse_args()
    maybe_mkdir(args.output_path)
    restrict_segmentations(args.mask_path, args.output_path)
