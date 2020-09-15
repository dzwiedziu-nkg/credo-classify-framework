import numpy as np

from numpy import ndarray

from credo_cf.classification.clustering.base_procedure import BaseProcedure


class SortedPreprocess(BaseProcedure):
    """
    Sort values in stack.
    """

    def procedure(self, stacked: ndarray):
        stacked_flat = stacked.reshape(len(stacked), -1)
        return np.sort(stacked_flat, axis=1)
