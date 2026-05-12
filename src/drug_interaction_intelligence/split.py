import copy
import random

import numpy as np


def collect_link_positions(drug_drug_matrix):
    link_positions = []
    non_link_positions = []
    for i in range(0, len(drug_drug_matrix)):
        for j in range(i + 1, len(drug_drug_matrix)):
            if drug_drug_matrix[i, j] == 1:
                link_positions.append([i, j])
            else:
                non_link_positions.append([i, j])
    return np.asarray(link_positions), non_link_positions


def holdout_by_link(drug_drug_matrix, ratio, seed):
    link_positions, non_link_positions = collect_link_positions(drug_drug_matrix)
    link_number = len(link_positions)

    random.seed(seed)
    index = np.arange(0, link_number)
    random.shuffle(index)

    test_index = index[0:int(link_number * ratio)]
    test_index.sort()
    test_link_positions = link_positions[test_index]
    train_matrix = copy.deepcopy(drug_drug_matrix)

    for row, col in test_link_positions:
        train_matrix[row, col] = 0
        train_matrix[col, row] = 0

    test_positions = list(test_link_positions) + list(non_link_positions)
    return train_matrix, test_positions
