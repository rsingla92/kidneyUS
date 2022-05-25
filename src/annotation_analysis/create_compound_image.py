import os
import cv2
import json
import argparse
import numpy as np 
import matplotlib.pyplot as plt

def display_multiple_img(images, filename, rows = 1, cols=1):
    figure, ax = plt.subplots(nrows=rows,ncols=cols )
    for ind,title in enumerate(images):
        ax.ravel()[ind].imshow(images[title])
        ax.ravel()[ind].set_title(title)
        ax.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()               

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGG csv to json converter")
    parser.add_argument("path1")
    parser.add_argument("path2")
    parser.add_argument("path3")
    parser.add_argument("savepath")
    args = parser.parse_args()
    for filename in os.listdir(args.path3):
        file1 = os.path.join(args.path1, filename)
        file2 = os.path.join(args.path2, filename)
        file3 = os.path.join(args.path3, filename)
        outfile = os.path.join(args.savepath, filename)
        img1 = cv2.imread(file1)
        print(file1)
        if len(img1.shape) < 3:
            img1 = np.repeat(img1[:, :, np.newaxis], 3, axis=2)
        img2 = cv2.imread(file2)
        img3 = cv2.imread(file3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = int(img1.shape[1]/2 - 75)
        scale = img1.shape[1] / 750
        cv2.putText(img1,'Original',(position, 50), font, scale,(255,255,255),2)
        cv2.putText(img2,'Jan',(position, 50), font, scale,(255,255,255),2)
        cv2.putText(img3,'Prediction',(position, 50), font, scale,(255,255,255),2)
        final = np.concatenate((img1, img2, img3), axis=1)
        cv2.imwrite(outfile, final)
        # display_multiple_img(images, outfile, 1, 3)
