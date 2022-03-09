from abc import ABC
from collections import defaultdict
from typing import List

from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

#from spherecluster import SphericalKMeans


class AbstractClustering(ABC):
    def __init__(self):
        self.method = None

    def fit(self, X) -> List[int]:
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
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

#
# class TaxoGenCluster(AbstractClustering):
#
#     def __init__(self, n_cluster):
#         super().__init__()
#         self.n_cluster = n_cluster
#         self.clus = SphericalKMeans(n_cluster)
#         self.clusters = defaultdict(list)  # cluster id -> members
#         self.membership = None  # a list contain the membership of the data points
#         self.center_ids = None  # a list contain the ids of the cluster centers
#         self.inertia_scores = None
#
#     def fit(self, X):
#         self.clus.fit(X)
#         labels = self.clus.labels_
#         for idx, label in enumerate(labels):
#             self.clusters[label].append(idx)
#         self.membership = labels
#         self.center_ids = self.gen_center_idx(X)
#         self.inertia_scores = self.clus.inertia_
#         print('Clustering concentration score:', self.inertia_scores)
#         return labels
#
#     # find the idx of each cluster center
#     def gen_center_idx(self, X):
#         ret = []
#         for cluster_id in range(self.n_cluster):
#             center_idx = self.find_center_idx_for_one_cluster(X, cluster_id)
#             ret.append((cluster_id, center_idx))
#         return ret
#
#     def find_center_idx_for_one_cluster(self, X, cluster_id):
#         query_vec = self.clus.cluster_centers_[cluster_id]
#         members = self.clusters[cluster_id]
#         best_similarity, ret = -1, -1
#         for member_idx in members:
#             member_vec = X[member_idx]
#             cosine_sim = self.calc_cosine(query_vec, member_vec)
#             if cosine_sim > best_similarity:
#                 best_similarity = cosine_sim
#                 ret = member_idx
#         return ret
#
#     def calc_cosine(self, vec_a, vec_b):
#         return 1 - cosine(vec_a, vec_b)
