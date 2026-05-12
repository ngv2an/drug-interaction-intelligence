import pytest

from drug_interaction_intelligence.data import load_drug_ids, load_matrix_csv, validate_square_matrix


def test_load_matrix_csv_shape():
    matrix = load_matrix_csv("dataset/drug_drug_matrix.csv", "int")
    assert matrix.shape == (548, 548)


def test_validate_square_matrix_accepts_dataset():
    matrix = load_matrix_csv("dataset/chem_Jacarrd_sim.csv", "float")
    validate_square_matrix(matrix, "chem_similarity")


def test_validate_square_matrix_rejects_non_square_matrix():
    with pytest.raises(ValueError):
        validate_square_matrix([[1, 2, 3], [4, 5, 6]], "bad_matrix")


def test_load_drug_ids():
    drug_ids = load_drug_ids()
    assert len(drug_ids) == 548
    assert drug_ids[0].startswith("DB")
