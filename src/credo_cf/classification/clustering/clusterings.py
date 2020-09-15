from numpy import ndarray
from sklearn.cluster import KMeans

from credo_cf.classification.clustering.base_procedure import BaseProcedure

DEFAULT_KMEANS_CONFIG = {
    'n_clusters': 20,
    'n_init': 20,
    'max_iter': 300,
    'random_state': 0,
    'verbose': 1,
    'algorithm': 'elkan'
}


class KMeansClassify(BaseProcedure):

    def __init__(self, config=None) -> None:
        _config = {} if config is None else config
        self.config = {
            **DEFAULT_KMEANS_CONFIG,
            **_config
        }

    def procedure(self, stack: ndarray) -> KMeans:
        kmeans = KMeans(**self.config)
        kmeans.fit(stack)
        return kmeans
