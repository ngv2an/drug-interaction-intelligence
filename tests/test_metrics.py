import numpy as np

from drug_interaction_intelligence.metrics import collect_position_scores, metrics_list_to_dict


def test_metrics_list_to_dict():
    metrics = metrics_list_to_dict(["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"])
    assert metrics["auc"] == 0.1
    assert metrics["f1"] == 0.6


def test_collect_position_scores():
    real_matrix = np.asarray([[0, 1], [1, 0]])
    predict_matrix = np.asarray([[0.0, 0.8], [0.8, 0.0]])
    labels, scores = collect_position_scores(real_matrix, predict_matrix, [[0, 1]])

    assert labels.tolist() == [1]
    assert scores.tolist() == [0.8]
