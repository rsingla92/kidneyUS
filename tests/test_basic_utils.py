import os
import csv
import tempfile

from tools.basic_utils import maybe_mkdir, write_rows_to_file, format_floats_for_csv


def test_maybe_mkdir_creates_directory(tmp_path):
    new_dir = tmp_path / "test_dir"
    maybe_mkdir(str(new_dir))
    assert new_dir.is_dir()


def test_maybe_mkdir_existing_directory(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    maybe_mkdir(str(existing))
    assert existing.is_dir()


def test_write_rows_to_file(tmp_path):
    outfile = str(tmp_path / "out.csv")
    rows = [["a", "b", "c"], [1, 2, 3], [4, 5, 6]]
    write_rows_to_file(outfile, rows)

    with open(outfile, "r") as f:
        reader = csv.reader(f)
        result = list(reader)
    assert len(result) == 3
    assert result[0] == ["a", "b", "c"]
    assert result[1] == ["1", "2", "3"]


def test_format_floats_for_csv():
    result = format_floats_for_csv([1.12345, 2.6789, 3.0])
    assert result == [1.12, 2.68, 3.0]


def test_format_floats_for_csv_empty():
    assert format_floats_for_csv([]) == []
