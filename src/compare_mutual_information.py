import os
import argparse
import numpy as np
import nibabel as nib

def find_mutual_information(img, stats):
    hist_2d, x_edges, y_edges = np.histogram2d(
        img.ravel(),
        stats.ravel(),
        bins=25)
    return mutual_information(hist_2d)

def normalize_minmax_data(image_data,min_val=1,max_val=99):
        min_val_1p=np.percentile(image_data,min_val)
        max_val_99p=np.percentile(image_data,max_val)
        final_image_data=np.zeros((image_data.shape[0],image_data.shape[1]), dtype=np.float64)
        # min-max norm on total 3D volume
        final_image_data=(image_data-min_val_1p)/(max_val_99p-min_val_1p)
        return final_image_data

def mutual_information(hgram):
     """ Mutual information for joint histogram
     """
     # Convert bins counts to probability values
     pxy = hgram / float(np.sum(hgram))
     px = np.sum(pxy, axis=1) # marginal for x over y
     py = np.sum(pxy, axis=0) # marginal for y over x
     px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
     # Now we can do the calculation using the pxy, px_py 2D arrays
     nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
     return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find average mutual information of stats with image")
    parser.add_argument("img_dir", help="Path to images")
    parser.add_argument("stats_dir", help="Path to stats")
    args = parser.parse_args()
    img_dir = args.img_dir
    stats_dir = args.stats_dir

    shape_total = 0
    scale_total = 0
    count = 0
    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        stats_file = img_file.replace(".nii.gz", ".npz")
        stats_path = os.path.join(stats_dir, stats_file)
        stats_load = np.load(stats_path)
        stats = stats_load['arr_0']
        
        img_load = nib.load(img_path)
        img = img_load.get_data()
        img = np.squeeze(img)
        shape_mi = find_mutual_information(img, normalize_minmax_data(stats[0,:,:]))
        scale_mi = find_mutual_information(img, normalize_minmax_data(stats[1,:,:]))
        shape_total += shape_mi
        scale_total += scale_mi
        count += 1
        print(count)

    print(shape_total / count, scale_total / count)