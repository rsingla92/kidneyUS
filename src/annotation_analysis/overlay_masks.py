import os
import argparse
import sys

sys.path.append('../')

from tools.image_utils import *
from tools.basic_utils import maybe_mkdir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create overlays")
    parser.add_argument("img_path")
    parser.add_argument("mask_path")
    parser.add_argument("overlay_path")
    args = parser.parse_args()

    maybe_mkdir(args.overlay_path)

    create_all_overlays(args.img_path, args.mask_path, args.overlay_path)