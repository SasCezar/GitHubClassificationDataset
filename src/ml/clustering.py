from abc import ABC
from typing import List

import numpy
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture


class AbstractClustering(ABC):
    def __init__(self):
        self.method = None

    def fit(self, X) -> List[int]:
        return self.method.fit_predict(X)


class GMMClustering(AbstractClustering):
    def __init__(self, **kwargs):
        super().__init__()
        self.method = GaussianMixture(**kwargs)


class KMeansClustering(AbstractClustering):
    def __init__(self, **kwargs):
        super().__init__()
        self.method = KMeans(**kwargs)


class DBSCANClustering(AbstractClustering):
    def __init__(self, **kwargs):
        super().__init__()
        self.method = DBSCAN(**kwargs)
