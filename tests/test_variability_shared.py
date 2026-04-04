import csv
import numpy as np

from tools.variability_shared import (
    get_conversions, generate_masks, compute_dsc,
    compute_precision, compute_recall
)


def test_get_conversions(tmp_path):
    csv_file = str(tmp_path / "conv.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "mm/pixel"])
        writer.writerow(["image.png", "0.5"])
        writer.writerow(["image2.png", "1.0"])

    result = get_conversions(csv_file)
    assert result["image.png"] == "0.5"
    assert result["image2.png"] == "1.0"


def test_generate_masks_capsule():
    cap = np.zeros((10, 10, 3), dtype=np.uint8)
    cap[3:7, 3:7, :] = 1
    reg = np.zeros((10, 10, 3), dtype=np.uint8)
    masks = generate_masks([cap], [reg], cls=1, num=1)
    assert len(masks) == 1
    assert masks[0].sum() > 0


def test_generate_masks_region():
    cap = np.zeros((10, 10, 3), dtype=np.uint8)
    reg = np.zeros((10, 10, 3), dtype=np.uint8)
    reg[3:7, 3:7, :] = 2
    masks = generate_masks([cap], [reg], cls=3, num=1)
    assert len(masks) == 1
    assert masks[0].sum() > 0


def test_compute_dsc_perfect_overlap():
    mask = np.ones((10, 10, 3), dtype=bool)
    assert compute_dsc(mask, mask) == 1.0


def test_compute_dsc_no_overlap():
    mask1 = np.zeros((10, 10, 3), dtype=bool)
    mask2 = np.zeros((10, 10, 3), dtype=bool)
    mask1[:5, :, :] = True
    mask2[5:, :, :] = True
    assert compute_dsc(mask1, mask2) == 0.0


def test_compute_dsc_partial_overlap():
    mask1 = np.zeros((10, 10, 3), dtype=bool)
    mask2 = np.zeros((10, 10, 3), dtype=bool)
    mask1[:7, :, :] = True
    mask2[3:, :, :] = True
    dsc = compute_dsc(mask1, mask2)
    assert 0 < dsc < 1


def test_compute_dsc_both_empty():
    mask = np.zeros((10, 10, 3), dtype=bool)
    assert compute_dsc(mask, mask) == 1.0


def test_compute_precision_perfect():
    pred = np.ones((10, 10), dtype=bool)
    truth = np.ones((10, 10), dtype=bool)
    assert compute_precision(pred, truth) == 1.0


def test_compute_precision_no_tp():
    pred = np.zeros((10, 10), dtype=bool)
    truth = np.ones((10, 10), dtype=bool)
    assert compute_precision(pred, truth) == 0.0


def test_compute_recall_perfect():
    pred = np.ones((10, 10), dtype=bool)
    truth = np.ones((10, 10), dtype=bool)
    assert compute_recall(pred, truth) == 1.0


def test_compute_recall_no_tp():
    pred = np.zeros((10, 10), dtype=bool)
    truth = np.ones((10, 10), dtype=bool)
    assert compute_recall(pred, truth) == 0.0
