"""Collection of simple useful functions
"""
import os
import csv


def maybe_mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def write_rows_to_file(csv_file, rows):
    """
    Write given rows of data to CSV file

    :param csv_file: path to output CSV files
    :param rows: rows of data to write to file
    """
    with open(csv_file, "w", newline='') as fp:
        csv_writer = csv.writer(fp)
        for row in rows:
            csv_writer.writerow(row)

def format_floats_for_csv(l):
    new_l = []
    for num in l:
        truncated_num = float("%.2f" % num)
        new_l.append(truncated_num)
    return new_l
