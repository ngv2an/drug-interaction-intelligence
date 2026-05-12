import numpy as np

from drug_interaction_intelligence.preprocessing import normalize_score_vector, to_1d_float_array


def test_to_1d_float_array_flattens_values():
    values = to_1d_float_array([[1], [2], [3]])
    assert values.tolist() == [1.0, 2.0, 3.0]


def test_normalize_score_vector_returns_1d_range():
    normalized = normalize_score_vector([10, 20, 30])
    assert normalized.shape == (3,)
    assert np.isclose(normalized.min(), 0.0)
    assert np.isclose(normalized.max(), 1.0)
