import csv

import numpy as np

from .paths import resolve_project_path


def load_matrix_csv(path, value_type="float"):
    """Load project matrix CSV files and ignore the first header row/ID column."""
    matrix_data = []
    converter = int if value_type == "int" else float
    with open(resolve_project_path(path), "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)
        for row_vector in csv_reader:
            matrix_data.append([converter(value) for value in row_vector[1:]])
    return np.asarray(matrix_data)


def validate_square_matrix(matrix, name):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("%s must be 2-dimensional" % name)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("%s must be square, got shape %s" % (name, matrix.shape))


def load_drug_ids(path="dataset/Mô tả về các đặc trưng/drug_list.txt"):
    drug_ids = []
    with open(resolve_project_path(path), "r") as drug_file:
        for line in drug_file:
            parts = line.strip().split()
            if len(parts) >= 2:
                drug_ids.append(parts[1])
    return drug_ids
