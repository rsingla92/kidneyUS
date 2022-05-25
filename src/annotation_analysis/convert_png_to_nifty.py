from skimage import io
import SimpleITK as sitk
import numpy as np
import os 
import random
import json
import argparse
from skimage.color import rgb2gray

def convert_2d_image_to_nifti(input_filename: str, output_filename_truncated: str, spacing=(999, 1, 1),
                              transform=None, is_grayscale: bool = False) -> None:
    img = io.imread(input_filename)
    img = rgb2gray(img)

    if transform is not None:
        img = transform(img)

    if len(img.shape) == 2:  # 2d image with no color channels
        img = img[None, None]  # add dimensions
    else:
        assert len(img.shape) == 3, "image should be 3d with color channel last but has shape %s" % str(img.shape)
        # we assume that the color channel is the last dimension. Transpose it to be in first
        img = img.transpose((2, 0, 1))
        # add third dimension
        img = img[:, None]

    # image is now (c, x, x, z) where x=1 since it's 2d
    if is_grayscale:
        assert img.shape[0] == 1, 'segmentations can only have one color channel, not sure what happened here'
    else:
        if(img.shape[0]==1):
            img = np.array([img[0], img[0], img[0]])

    for j, i in enumerate(img):

        if is_grayscale:
            i = i.astype(np.uint32)
        print(type(i))
        itk_img = sitk.GetImageFromArray(i)
        print(np.max(i))
        print(np.min(i))
        itk_img.SetSpacing(list(spacing)[::-1])
        print(itk_img.GetPixel(0, 0, 0))
        if not is_grayscale:
            sitk.WriteImage(itk_img, output_filename_truncated + "_%04.0d.nii.gz" % j)
        else:
            sitk.WriteImage(itk_img, output_filename_truncated + ".nii.gz")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="png to nifty converter")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    print(args.input_dir)
    for filename in os.listdir(args.input_dir):
        if filename.split(".")[-1] == "png":
            prefix = filename.split(".")[0]
            input_file = os.path.join(args.input_dir, filename)
            output_file = os.path.join(args.output_dir, filename[:-4])
            convert_2d_image_to_nifti(input_file, output_file, is_grayscale=True)
