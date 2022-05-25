import numpy as np
from skimage import io
import SimpleITK as sitk
import random
import os
import argparse

def convert_2d_segmentation_nifti_to_img(nifti_file: str, output_filename: str, transform=None, export_dtype=np.uint8):
    img = sitk.GetArrayFromImage(sitk.ReadImage(nifti_file))
    assert img.shape[0] == 1, "This function can only export 2D segmentations!"
    img = img[0]
    if transform is not None:
        img = transform(img)
    output_filename = output_filename.replace('_0000','')
    io.imsave(output_filename, img.astype(export_dtype), check_contrast=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="nifty to png converter")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    print(args.input_dir)
    for filename in os.listdir(args.input_dir):
        if filename.split(".")[1] == "nii":
            prefix = filename.split(".")[0]
            input_file = os.path.join(args.input_dir, filename)
            output_file = os.path.join(args.output_dir, filename).replace('.nii.gz','.png')
            convert_2d_segmentation_nifti_to_img(input_file, output_file)
