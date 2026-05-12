import numpy as np
from sklearn.preprocessing import MinMaxScaler


def to_1d_float_array(values):
    return np.real(np.asarray(values)).astype(float).ravel()


def normalize_score_vector(values):
    values = to_1d_float_array(values).reshape(-1, 1)
    return MinMaxScaler().fit_transform(values).ravel()
