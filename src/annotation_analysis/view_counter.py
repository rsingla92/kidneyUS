"""
Count number of images of each view

"""
import csv
import json
import argparse
import sys

sys.path.append('../')

from tools.variability_shared import get_repeats


def get_views(f):
    """
    Get view counts from given annotation file

    :param f: VGG annotations file
    :return: dictionary of views for each file
    """
    view_dict = {}
    with open(f, 'r') as csv_fp:
        csv_reader = csv.DictReader(csv_fp)
        for row in csv_reader:
            filename = row["filename"]
            view = row["file_attributes"]
            att_map = json.loads(view)
            view_dict[filename] = att_map["View"]
    return view_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get view statistics from data in VGG annotation file")
    parser.add_argument("repeats",
                        help="File containing sets of repeating images")
    parser.add_argument("infile", help="Annotation file")
    parser.add_argument("outfile", help="File for view statistics")

    args = parser.parse_args()
    repeats = get_repeats(args.repeats)
    views = get_views(args.infile)

    transverse = 0
    longitudinal = 0
    other = 0

    for f in views.keys():
        if f not in repeats:
            if views[f] == "Transverse":
                transverse += 1
            elif views[f] == "Longitudinal":
                longitudinal += 1
            else:
                other += 1
    total = transverse + longitudinal + other

    with open(args.outfile, "w") as fp:
        fp.write("KIDNEY VIEW STATISTICS\n")
        fp.write("Total: {:d}\n".format(total))
        fp.write("Num Transverse: {:d}\n".format(transverse))
        fp.write("Num Longitudinal: {:d}\n".format(longitudinal))
        fp.write("Num Other: {:d}\n".format(other))
        fp.write("% Transverse: {:.2%}\n".format(transverse/total))
        fp.write("% Longitudinal: {:.2%}\n".format(longitudinal/total))
        fp.write("% Other: {:.2%}\n".format(other/total))
