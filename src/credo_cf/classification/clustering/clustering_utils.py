from typing import List

import numpy as np


def make_stack(list_of_matrices: List[np.ndarray]) -> np.ndarray:
    """
    Join array of xD matrices to one (x+1)D matrix. I.e. list od 2D matrix with shape (a,b) join to 3D matrix with shape (len(list), a, b)
    :param list_of_matrices: list of matrices to join
    :return: stack of joined matrices
    """
    return np.vstack(list_of_matrices)
