"""Script which calculates Cohen's Kappa for quality scores from two annotators
and creates histogram of quality scores
"""
import csv
import json
import sys
import argparse
import matplotlib.pyplot as plt

sys.path.append('../')

from tools.variability_shared import get_repeats

POOR = "Poor"
FAIR = "Fair"
GOOD = "Good"
UNSATISFACTORY = "Unsatisfactory"
QUALITIES = {GOOD: 0, FAIR: 1, POOR: 2, UNSATISFACTORY: 3}


def get_qualities(f, repeats):
    """
    Get map of quality ratings from CSV file of annotations

    :param f: path to CSV annotations
    :return: map of filenames to quality ratings
    """
    quality_dict = {}
    with open(f, 'r') as csv_fp:
        csv_reader = csv.DictReader(csv_fp)
        for row in csv_reader:
            filename = row["filename"]
            if filename not in repeats:
                quality = row["file_attributes"]
                att_map = json.loads(quality)
                quality_dict[filename] = att_map["Quality"]
    return quality_dict


def count_qualities(q1, q2):
    """
    Get map of quality ratings from CSV file of annotations

    :param q1: first quality map
    :param q2: second quality map
    :return: list of total qualities from both annotators
    """
    quality_counts = [0, 0, 0, 0]
    for key in q1.keys():
        quality = q1[key]
        index = QUALITIES[quality]
        quality_counts[index] = quality_counts[index] + 1
    for key in q2.keys():
        quality = q2[key]
        index = QUALITIES[quality]
        quality_counts[index] = quality_counts[index] + 1
    return quality_counts


def calculate_po(q1, q2):
    """
    Calculate Po for Cohen's Kappa

    :param q1: first quality map
    :param q2: second quality map
    :return: Po
    """
    total = 0
    agreed = 0
    mismatched = []
    for f in q1.keys():
        total += 1
        if q1[f] == q2[f]:
            agreed += 1
        else:
            mismatched.append(f)
    return float(agreed) / total, mismatched


def get_proportions(q):
    """
    Get proportion of each quality score

    :param q: quality score map
    :return: map of proportions of each class
    """
    total = 0
    p_map = {GOOD: 0, POOR: 0, FAIR: 0, UNSATISFACTORY: 0}
    for f in q.keys():
        p_map[q[f]] = p_map[q[f]] + 1
        total += 1
    for key in p_map.keys():
        p_map[key] = p_map[key] / float(total)
    return p_map


def calculate_pe(p1, p2):
    """
    Calculate Pe for Cohen's Kappa

    :param p1: quality proportions for first annotator
    :param p2: quality proportions for second annotator
    :return: Pe
    """
    p_poor = p1[POOR] * p2[POOR]
    p_fair = p1[FAIR] * p2[FAIR]
    p_uns = p1[UNSATISFACTORY] * p2[UNSATISFACTORY]
    return p_poor + p_fair + p_uns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Get Cohen's Kappa for two csv files from VGG")
    parser.add_argument("file1", help="Path to first set of annotations")
    parser.add_argument("file2", help="Path to second set of annotations")
    parser.add_argument("repeat_file", help="File listing repeated annotations")
    parser.add_argument("outfile", help="File to save Cohen's Kappa in")
    parser.add_argument("histogram", help="Histogram file path")

    args = parser.parse_args()
    extra_files = get_repeats(args.repeat_file)
    quality_1 = get_qualities(args.file1, extra_files)
    quality_2 = get_qualities(args.file2, extra_files)
    po, mismatched = calculate_po(quality_1, quality_2)
    p1 = get_proportions(quality_1)
    p2 = get_proportions(quality_2)
    pe = calculate_pe(p1, p2)
    ck = (po-pe)/(1-pe)
    print(ck)
    with open(args.outfile, "w") as f:
        f.writelines(mismatched)

    quality_count = count_qualities(quality_1, quality_2)
    plt.bar([GOOD, FAIR, POOR, UNSATISFACTORY], quality_count)
    print([GOOD, FAIR, POOR, UNSATISFACTORY])
    # print(quality_count)
    print(quality_count)
    plt.title("Quality Scores, Cohen's Kappa=" + str(ck))
    plt.xlabel('Quality Score')
    plt.ylabel('Number of Frames')
    plt.savefig(args.histogram)
    plt.clf()
