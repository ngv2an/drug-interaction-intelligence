import numpy as np

from drug_interaction_intelligence.split import collect_link_positions, holdout_by_link


def test_collect_link_positions_counts_upper_triangle_only():
    matrix = np.asarray(
        [
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ]
    )
    link_positions, non_link_positions = collect_link_positions(matrix)

    assert link_positions.tolist() == [[0, 1], [0, 2]]
    assert non_link_positions == [[1, 2]]


def test_holdout_by_link_removes_test_edges_symmetrically():
    matrix = np.asarray(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]
    )

    train_matrix, test_positions = holdout_by_link(matrix, ratio=0.5, seed=0)
    removed_count = int((matrix.sum() - train_matrix.sum()) / 2)

    assert removed_count == 1
    assert train_matrix.tolist() == train_matrix.T.tolist()
    assert len(test_positions) > 0
