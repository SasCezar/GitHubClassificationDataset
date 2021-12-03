from abc import abstractmethod, ABC
from typing import Iterable, Callable, List, Tuple, Optional, Dict

import numpy
import numpy as np
from numpy import ndarray, array
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity

from src.ml.clustering import AbstractClustering
from src.ml.embeddings import AbstractEmbeddingModel


class Linking(ABC):
    def __init__(self, embedding: AbstractEmbeddingModel, threshold: float = 0.7):
        """
        :param threshold: Minimum threshold to link a term with another
        :param embedding: Method to create the embeddings of the words
        """
        self.threshold = threshold
        self.embedding = embedding
        self.similarity = cosine_similarity

    def run(self, ranking: DataFrame) -> List[Tuple[int, int]]:
        embeddings = self.get_embeddings(ranking['q_id'])
        res = self.create_taxonomy(ranking, embeddings)

        return res

    @abstractmethod
    def create_taxonomy(self, ranking: DataFrame, embeddings: ndarray) -> List[Tuple[int, int]]:
        """
        Links elements in a ranked list to create a taxonomy
        :param ranking:
        :param embeddings:
        :return:
        """
        pass

    def get_embeddings(self, topics: Iterable) -> array:
        res = []
        for topic in topics:
            res.append(list(self.embedding.get_embedding(topic)))

        return numpy.array(res)

    def compute_similarity(self, X, Y: Optional = None) -> ndarray:
        if Y is not None:
            return self.similarity(X, Y)

        return self.similarity(X)


class OrderLinking(Linking):
    """
    Creates a hierarchy based on the order, an item can only be linked to elements that are before him in the ranking
    """

    def create_taxonomy(self, ranking: DataFrame, embeddings: ndarray) -> List[Tuple[int, int]]:
        similarity = self.compute_similarity(embeddings)
        order = sorted([(i, x) for i, x in enumerate(ranking['mean'])], key=lambda x: x[1], reverse=True)
        remap = {k: i for i, (k, _) in enumerate(order)}
        topics = ranking['topic'].tolist()
        res = []
        for i in range(len(similarity)):
            most_similar = similarity[i].argsort()[::-1]
            for ms in most_similar:
                if remap[i] > ms != i and similarity[i][ms] >= self.threshold:
                    res.append((topics[i], topics[ms]))
                    break

            # res.append((i, -1))

        return res


class ClusterLinking(Linking):
    """
    Creates a hierarchy based on clusters, an item is linked to the best element in a cluster that has elements with
    higher ranking (higher cluster)
    """

    def __init__(self, embedding: AbstractEmbeddingModel, clustering: AbstractClustering, threshold: float = 0.0):
        super().__init__(embedding, threshold)
        self.clustering = clustering

    def create_taxonomy(self, ranking: DataFrame, embeddings) -> List[Tuple[int, int]]:
        n, clusters = self.cluster(ranking)
        topics = ranking['topic'].tolist()
        res = []
        pairs = list(zip(range(0, n - 1), range(1, n)))
        for i, j in pairs:
            Xi, remap_i = self.submatrix(embeddings, [k for k, x in enumerate(clusters) if x == i])
            Xj, remap_j = self.submatrix(embeddings, [k for k, x in enumerate(clusters) if x == j])
            similarity = self.compute_similarity(Xi, Xj)
            for r in range(len(similarity)):
                most_similar = np.argmax(similarity[r])
                if similarity[r][most_similar] >= self.threshold:
                    res.append((topics[remap_i[r]], topics[remap_j[most_similar]]))
                else:
                    res.append((remap_i[r], -1))

        return res

    def cluster(self, ranking: DataFrame):
        """
        Cluster the ranking
        :param ranking:
        :return:
        """
        clusters = self.clustering.fit(ranking['mean'])
        ranking['cluster'] = clusters
        ordered_clusters = self.order_clusters(ranking)
        return len(set(clusters)), ordered_clusters

    @staticmethod
    def submatrix(embeddings: ndarray, cluster: List) -> Tuple[ndarray, Dict[int, int]]:
        remap = {i: c for i, c in enumerate(cluster)}
        X = numpy.take(embeddings, cluster, 0)

        return X, remap

    @staticmethod
    def order_clusters(ranking: DataFrame) -> List[int]:
        cluster_mean = ranking.groupby('cluster').agg({'mean': 'mean'})
        df: DataFrame = cluster_mean.sort_values(['mean'], ascending=True).reset_index()
        remap = df.to_dict()['cluster']
        res = [remap[i] for i in ranking['cluster']]
        return res
