import os
import numpy as np
import cv2

from tools.image_utils import create_compound_image, create_overlay


def test_create_compound_image(tmp_path):
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.full((100, 100, 3), 128, dtype=np.uint8)

    f1 = str(tmp_path / "img1.png")
    f2 = str(tmp_path / "img2.png")
    outfile = str(tmp_path / "compound.png")

    cv2.imwrite(f1, img1)
    cv2.imwrite(f2, img2)

    create_compound_image([f1, f1, f2], ["skip", "Image 1", "Image 2"], outfile)

    assert os.path.exists(outfile)
    result = cv2.imread(outfile)
    assert result.shape[0] == 100
    assert result.shape[1] == 200


def test_create_overlay(tmp_path):
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    cap_mask = np.zeros((50, 50, 3), dtype=np.uint8)
    cap_mask[10:40, 10:40, :] = 1
    reg_mask = np.zeros((50, 50, 3), dtype=np.uint8)
    reg_mask[15:35, 15:35, :] = 2

    img_f = str(tmp_path / "img.png")
    cap_f = str(tmp_path / "cap.png")
    reg_f = str(tmp_path / "reg.png")
    out_f = str(tmp_path / "overlay.png")

    cv2.imwrite(img_f, img)
    cv2.imwrite(cap_f, cap_mask)
    cv2.imwrite(reg_f, reg_mask)

    create_overlay(img_f, cap_f, reg_f, out_f)

    assert os.path.exists(out_f)
    result = cv2.imread(out_f)
    assert result.shape == img.shape


def test_create_overlay_missing_image(tmp_path):
    out_f = str(tmp_path / "overlay.png")
    create_overlay("nonexistent.png", "cap.png", "reg.png", out_f)
    assert not os.path.exists(out_f)
