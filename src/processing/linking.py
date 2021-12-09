from abc import abstractmethod, ABC
from typing import Iterable, List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
import numpy
import numpy as np
from numpy import ndarray, array
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity

from src.ml.clustering import AbstractClustering
from src.ml.embeddings import AbstractEmbeddingModel


class Linking(ABC):
    def __init__(self, embedding: AbstractEmbeddingModel, embedding_key, threshold: float = 0.0, plot=True):
        """
        :param threshold: Minimum threshold to link a term with another
        :param embedding: Method to create the embeddings of the words
        """
        self.threshold = threshold
        self.embedding = embedding
        self.similarity = cosine_similarity
        self.plot = plot
        self.embedding_key = embedding_key

    def run(self, ranking: DataFrame) -> List[Tuple[int, int]]:
        ranking = ranking.sort_values('mean', ascending=False)
        ranking['embeddings'] = self.get_embeddings(ranking[self.embedding_key])
        res = self.create_taxonomy(ranking)

        return res

    @abstractmethod
    def create_taxonomy(self, ranking: DataFrame) -> List[Tuple[int, int]]:
        """
        Links elements in a ranked list to create a taxonomy
        :param ranking:
        :return:
        """
        pass

    def get_embeddings(self, topics: Iterable) -> array:
        res = []
        for topic in topics:
            res.append(list(self.embedding.get_embedding(topic)))

        return res

    def compute_similarity(self, X, Y: Optional = None) -> ndarray:
        if Y is not None:
            return self.similarity(X, Y)

        return self.similarity(X)

    def _plot(self, mean, most_similar, i):
        ms_mean = [abs(mean[n] - mean[i]) for n in most_similar]
        plt.plot(list(range(len(ms_mean))), ms_mean)
        plt.show()


class OrderLinking(Linking):
    """
    Creates a hierarchy based on the order, an item can only be linked to elements that are before him in the ranking
    """

    def create_taxonomy(self, ranking: DataFrame) -> List[Tuple[int, int]]:
        embeddings = ranking['embeddings'].to_list()
        similarity = self.compute_similarity(embeddings)
        topics = ranking['topic'].tolist()
        res = []
        for i in range(len(similarity)):
            most_similar = similarity[i].argsort()[::-1]
            for ms in most_similar:
                if i < ms and similarity[i][ms] >= self.threshold:
                    res.append((topics[i], topics[ms]))
                    break

        return res


class OrderDistanceLinking(Linking):
    """
    Creates a hierarchy based on the order, an item can only be linked to elements that are before it in the ranking
    """

    def create_taxonomy(self, ranking: DataFrame) -> List[Tuple[int, int]]:
        similarity = self.compute_similarity(ranking['embeddings'])
        topics = ranking['topic'].tolist()
        mean = ranking['mean'].tolist()
        std = ranking['std'].tolist()
        res = []
        for i in range(len(similarity)):
            most_similar = similarity[i].argsort()[::-1]
            self._plot(mean, most_similar, i) if self.plot else None
            for ms in most_similar:
                if i > ms and similarity[i][ms] >= self.threshold and mean[i] < mean[i] + std[i]:
                    res.append((topics[i], topics[ms]))
                    break

        return res


class ClusterLinking(Linking, ABC):
    def __init__(self, embedding: AbstractEmbeddingModel, embedding_key: str, clustering: AbstractClustering,
                 clustering_key: str = 'mean', threshold: float = 0.0):
        super().__init__(embedding, embedding_key, threshold)
        self.clustering = clustering
        self.clustering_key = clustering_key

    def cluster(self, ranking: DataFrame, order=True):
        """
        Cluster the ranking
        :param ranking:
        :param order:
        :return:
        """
        clusters = self.clustering.fit(ranking[self.clustering_key].to_numpy().reshape(-1, 1))
        ranking['cluster'] = clusters
        ordered_clusters = self.order_clusters(ranking) if order else clusters
        return len(set(clusters)), ordered_clusters

    @staticmethod
    def order_clusters(ranking: DataFrame) -> List[int]:
        cluster_mean = ranking.groupby('cluster').agg({'mean': 'mean'})
        df: DataFrame = cluster_mean.sort_values(['mean'], ascending=True).reset_index()
        remap = df.to_dict()['cluster']
        res = [remap[i] for i in ranking['cluster']]
        return res


class RankingClusterLinking(ClusterLinking):
    """
    Creates a hierarchy based on clusters, an item is linked to the best element in a cluster that has elements with
    higher ranking (higher cluster)
    """

    def create_taxonomy(self, ranking: DataFrame) -> List[Tuple[int, int]]:
        n, clusters = self.cluster(ranking, order=True)
        topics = ranking['topic'].tolist()
        res = []
        pairs = list(zip(range(0, n - 1), range(1, n)))
        for i, j in pairs:
            Xi, remap_i = self.submatrix(ranking['embeddings'], [k for k, x in enumerate(clusters) if x == i])
            Xj, remap_j = self.submatrix(ranking['embeddings'], [k for k, x in enumerate(clusters) if x >= j < x + 2])
            similarity = self.compute_similarity(Xi, Xj)
            for r in range(len(similarity)):
                most_similar = np.argmax(similarity[r])
                if similarity[r][most_similar] >= self.threshold:
                    res.append((topics[remap_i[r]], topics[remap_j[most_similar]]))
                else:
                    res.append((topics[remap_i[r]], -1))

        return res

    @staticmethod
    def submatrix(embeddings: ndarray, cluster: List) -> Tuple[ndarray, Dict[int, int]]:
        remap = {i: c for i, c in enumerate(cluster)}
        X = numpy.take(embeddings, cluster, 0)

        return X, remap


class SemanticClusterLinking(ClusterLinking):
    """
    """
    def create_taxonomy(self, ranking: DataFrame) -> List[Tuple[int, int]]:
        _, clusters = self.cluster(ranking, order=False)
        pass