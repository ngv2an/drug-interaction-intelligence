import numpy as np


METRIC_NAMES = ["auc", "aupr", "precision", "recall", "accuracy", "f1"]


def metrics_list_to_dict(results):
    return {METRIC_NAMES[index]: float(results[index]) for index in range(0, len(METRIC_NAMES))}


def collect_position_scores(real_matrix, predict_matrix, test_positions):
    real_labels = []
    predicted_probability = []
    for row, col in test_positions:
        real_labels.append(real_matrix[row, col])
        predicted_probability.append(predict_matrix[row, col])
    return np.asarray(real_labels).ravel(), np.asarray(predicted_probability).ravel()
