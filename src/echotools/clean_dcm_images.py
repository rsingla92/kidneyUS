from glob import glob
from dicom import Dicom
import cv2
import os
import argparse
import pandas as pd
import numpy as np
from image import to_grayscale
import SimpleITK as sitk

patient_map = {}


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def maybe_mkdir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def number_image(filename, name):
    if name not in patient_map:
        patient_map[name] = len(patient_map)
    return str(patient_map[name]) + "_" + filename


def mask_and_convert_to_png(dcm_path, args, filename):
    """
    Masks and saves dicom at given path to new anonymized png file

    :param dcm_path: path to DICOM file to process
    :param args: script arguments
    :param filename: filename to save to
    """
    dicom = None
    filenames = []
    included_path = os.path.join(args.savepath, 'included')
    excluded_path = os.path.join(args.savepath, 'excluded')
    try:
        dicom = Dicom(dcm_path)
    except Exception as e:
        with open(args.errorfiles, "a+") as fp:
            message = dcm_path + ": " + str(e) + "\n"
            fp.write(message)
            print(message)
        return None, filenames
    if not dicom.check_contains_pixel_data():
        return None, filenames
    dicom.convert_colourspace()
    dicom.remove_patient_info()
    metadata = dicom.metadata()
    filename = number_image(filename, metadata['patient_name'])
    outpath = os.path.join(included_path, filename)
    try:
        # masked = dicom.masked_video(crop=args.crop, resized=False, grayscale=args.grayscale,
        #                             exclude_doppler=args.exclude_doppler)
        masked = dicom.video
    except (ValueError, RuntimeError, AttributeError, IndexError) as ve:
        with open(args.errorfiles, "a+") as fp:
            print(args.errorfiles)
            message = dcm_path + ": " + str(ve) + "\n"
            fp.write(message)
            print(message)
        masked = dicom.video
        outpath = os.path.join(excluded_path, filename)
        if args.grayscale:
             masked = to_grayscale(dicom.video)
        print("Fail")
    if dicom.is_video:
        for i, frame in enumerate(masked):
            if not args.grayscale:
                mask = cv2.cvtColor(frame.astype('uint8'), cv2.COLOR_RGB2BGR)
            else:
                mask = frame
            outfile = outpath.replace(".dcm", "_frame{}_anon.png").format(i)
            # cv2.imwrite(outfile, mask)
            filenames.append(os.path.basename(outfile))
    else:
        if not args.grayscale:
            masked = cv2.cvtColor(masked.astype('uint8'), cv2.COLOR_RGB2BGR)
        outfile = outpath.replace(".dcm", "_anon.png")
        # cv2.imwrite(outfile, masked)
        filenames.append(os.path.basename(outfile))
    return metadata, filenames


def mask_and_save_to_dicom(dcm_path, args, filename):
    """
    Masks and saves dicom at given path to new anonymized DICOM file

    :param dcm_path: path to DICOM file to process
    :param args: script arguments
    :param filename: filename to save to
    """
    dicom = Dicom(dcm_path)
    metadata = dicom.metadata()
    included_path = os.path.join(args.savepath, 'included')
    excluded_path = os.path.join(args.savepath, 'excluded')
    filename = number_image(filename, metadata['patient_name'])
    outpath = os.path.join(included_path, filename)
    try:
        dicom.mask_pixel_array(crop=args.crop, resized=False,
                               grayscale=args.grayscale,
                               exclude_doppler=args.exclude_doppler)
    except (ValueError, RuntimeError, AttributeError, IndexError) as ve:
        with open(args.errorfiles, "a+") as fp:
            fp.write(dcm_path + ": " + str(ve) + "\n")
        outpath = os.path.join(excluded_path, filename)
    dicom.anonymize()
    dicom.save(outpath)
    return metadata, [os.path.basename(outpath)]


def mask_and_save_to_nii(dcm_path, args, filename):
    """
    Masks and saves dicom at given path to new anonymized nifty file

    :param dcm_path: path to DICOM file to process
    :param args: script arguments
    :param filename: filename to save to
    """
    dicom = None
    filenames = []
    included_path = os.path.join(args.savepath, 'included')
    excluded_path = os.path.join(args.savepath, 'excluded')
    dicom = Dicom(dcm_path)
    dicom.convert_colourspace()
    dicom.remove_patient_info()
    metadata = dicom.metadata()
    # filename = number_image(filename, metadata['patient_name'])
    outpath = os.path.join(included_path, filename)

    if dicom.is_video:
        print(filename)

    if args.length=="video" and not dicom.is_video:
        return None, filenames
        # outpath = os.path.join(excluded_path, filename)
    if args.length=="img" and dicom.is_video:
        return None, filenames
        # outpath = os.path.join(excluded_path, filename)
    if not dicom.check_contains_pixel_data():
        return None, filenames
    try:
        masked = dicom.masked_video(crop=args.crop, resized=False, grayscale=args.grayscale,
                                    exclude_doppler=args.exclude_doppler)
    except (ValueError, RuntimeError, AttributeError, IndexError, OSError) as ve:
        with open(args.errorfiles, "a+") as fp:
            message = dcm_path + ": " + str(ve) + "\n"
            fp.write(message)
            print(message)
        masked = dicom.video
        outpath = os.path.join(excluded_path, filename)
        if args.grayscale:
             masked = to_grayscale(dicom.video)

    if not args.grayscale:
        masked = cv2.cvtColor(masked.astype('uint8'), cv2.COLOR_RGB2BGR)
    outfile = outpath.replace(".dcm", ".nii.gz")
    itk_img = sitk.GetImageFromArray(masked)
    filenames.append(os.path.basename(outfile))
    sitk.WriteImage(itk_img, outfile)
    return metadata, [os.path.basename(outpath)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCM file cleaner")
    parser.add_argument("filepath")
    parser.add_argument('--crop', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--grayscale', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--type', type=str, nargs='?', const=True, default="png")
    parser.add_argument('--exclude_doppler', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--save_metadata', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--length', type=str, nargs='?', const=True, default="both")
    parser.add_argument("savepath")
    parser.add_argument("errorfiles")
    args = parser.parse_args()

    image_data = []

    maybe_mkdir(args.savepath)
    maybe_mkdir(os.path.join(args.savepath, 'included'))
    maybe_mkdir(os.path.join(args.savepath, 'excluded'))

    failed_files = []
    for root, dirs, files in os.walk(args.filepath):
        for filename in files:
            # print(filename)
            if filename.split(".")[-1] == "dcm":
                dcm_path = os.path.join(root, filename)
                if args.type=="png":
                    metadata, filenames = mask_and_convert_to_png(dcm_path, args, filename)
                elif args.type=="nii":
                    metadata, filenames = mask_and_save_to_nii(dcm_path, args, filename)
                elif args.type=="dicom":
                    metadata, filenames = mask_and_save_to_dicom(dcm_path, args, filename)
                if metadata:
                    for f in filenames:
                        image_data.append([f,
                                           metadata['phn'],
                                           metadata['patient_name'],
                                           metadata['mrn'],
                                           metadata['dob'],
                                           metadata['sex'],
                                           metadata['study_date'],
                                           metadata['manufacturer'],
                                           metadata['model'],
                                           metadata['physical_delta_x'],
                                           metadata['physical_delta_y'],
                                           metadata['transducer_frequency'],
                                           dcm_path])
                else:
                    failed_files.append(filename)
    img_df = pd.DataFrame(image_data, columns=['filepath', 'phn',
                                               'patient_name', 'mrn', 'dob',
                                               'sex', 'study_date',
                                               'manufacturer', 'model',
                                               'physical_delta_x',
                                               'physical_delta_y', 
                                               'transducer_frequency',
                                               'dcm_path'])

    if args.save_metadata:
        image_df_path = os.path.join(args.savepath, "image_data.pkl")
        img_df.to_pickle(image_df_path)
        img_df.to_csv(os.path.join(args.savepath, "image_data.csv"))
