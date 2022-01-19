from abc import abstractmethod, ABC
from typing import Iterable, List, Tuple, Optional, Dict

import numpy
import numpy as np
from numpy import ndarray, array
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity

from src.ml.clustering import AbstractClustering
from src.ml.embeddings import AbstractEmbeddingModel

from wikidata.client import Client


class AbstractLinking(ABC):
    def __init__(self, embedding: AbstractEmbeddingModel, embedding_key, threshold: float = 0.0, **kwargs):
        """
        :param threshold: Minimum threshold to link a term with another
        :param embedding: Method to create the embeddings of the words
        """
        self.threshold = threshold
        self.embedding = embedding
        self.similarity = cosine_similarity
        self.embedding_key = embedding_key

    def run(self, ranking: DataFrame) -> Tuple[List[Tuple[str, str, int]], DataFrame]:
        ranking = ranking.sort_values('mean', ascending=False).reset_index(drop=True)
        ranking['embeddings'] = self.get_embeddings(ranking[self.embedding_key])
        res, ranking = self.create_taxonomy(ranking)

        return res, ranking

    @abstractmethod
    def create_taxonomy(self, ranking: DataFrame) -> Tuple[List[Tuple[str, str, int]], DataFrame]:
        """
        Links elements in a ranked list to create a taxonomy
        :param ranking:
        :return:
        """
        pass

    def get_embeddings(self, topics: Iterable) -> array:
        res = []
        topics = topics.to_records(index=False) if isinstance(topics, DataFrame) else topics
        for topic in topics:
            res.append(list(self.embedding.get_embedding(topic)))

        return res

    def compute_similarity(self, X, Y: Optional = None) -> ndarray:
        if Y is not None:
            return self.similarity(X, Y)

        return self.similarity(X)

    def update_threshold(self, depth):
        pass


class OrderLinking(AbstractLinking):
    """
    Creates a hierarchy based on the order, an item can only be linked to elements that are before him in the ranking
    """

    def create_taxonomy(self, ranking: DataFrame) -> Tuple[List[Tuple[str, str, int]], DataFrame]:
        embeddings = ranking['embeddings'].to_list()
        similarity = self.compute_similarity(embeddings)
        topics = ranking['topic'].tolist()
        res = []
        for i in range(len(similarity)):
            most_similar = similarity[i].argsort()[::-1]
            for ms in most_similar:
                if i < ms and self.threshold <= similarity[i][ms]:
                    res.append((topics[i], topics[ms], similarity[i][ms]))
                    break

        return res, ranking


class ClusterLinking(AbstractLinking, ABC):
    def __init__(self, embedding: AbstractEmbeddingModel, embedding_key: str, clustering: AbstractClustering,
                 clustering_key: str = 'mean', threshold: float = 0.0, decay=True, **kwargs):
        super().__init__(embedding, embedding_key, threshold)
        self.clustering = clustering
        self.clustering_key = clustering_key
        self.decay = decay

    def cluster(self, ranking: DataFrame, order=True):
        """
        Cluster the ranking
        :param ranking:
        :param order:
        :return:
        """
        clusters = self.clustering.fit(np.array(ranking[self.clustering_key].to_list()))
        ordered_clusters = self.order_clusters(clusters) if order else clusters
        ranking['cluster'] = ordered_clusters
        return len(set(clusters)), ordered_clusters

    @staticmethod
    def order_clusters(cluster: List) -> List[int]:
        seen = {}
        n = len(set(cluster))
        res = []
        for i in cluster:
            if i not in seen:
                seen[i] = n - (len(seen) + 1)
            res.append(seen[i])

        return res


class RankingClusterLinking(ClusterLinking):
    """
    Creates a hierarchy based on clusters, an item is linked to the best element in a cluster that has elements with
    higher ranking (higher cluster)
    """

    def create_taxonomy(self, ranking: DataFrame) -> Tuple[List[Tuple[str, str, int]], DataFrame]:
        n, clusters = self.cluster(ranking, order=True)
        topics = ranking['topic'].tolist()
        res = []
        pairs = list(zip(range(0, n - 2), range(1, n - 1)))
        for i, j in pairs:
            Xi, remap_i = self.submatrix(ranking['embeddings'], [k for k, x in enumerate(clusters) if x == i])
            l = [k for k, x in enumerate(clusters) if j < x < j + 4]
            Xj, remap_j = self.submatrix(ranking['embeddings'], l)
            similarity = self.compute_similarity(Xi, Xj)
            self.update_threshold(j) if self.decay else None

            for r in range(len(similarity)):
                most_similar = np.argmax(similarity[r])
                if similarity[r][most_similar] >= self.threshold:
                    res.append((topics[remap_i[r]], topics[remap_j[most_similar]], similarity[r][most_similar]))
                else:
                    res.append((topics[remap_i[r]], -1, 0))

        return res, ranking

    @staticmethod
    def submatrix(embeddings: ndarray, cluster: List) -> Tuple[ndarray, Dict[int, int]]:
        remap = {i: c for i, c in enumerate(cluster)}
        X = numpy.take(embeddings, cluster, 0).tolist()

        return X, remap

    def update_threshold(self, j):
        self.threshold = (1 / (1 + 0.02 * j)) * self.threshold


class ReuseRankingClusterLinking(RankingClusterLinking):
    def create_taxonomy(self, ranking: DataFrame) -> Tuple[List[Tuple[str, str, int]], DataFrame]:
        n, clusters = self.cluster(ranking, order=True)
        topics = ranking['topic'].tolist()
        res = []
        pairs = list(zip(range(0, n - 2), range(1, n - 1)))
        skipped = set()
        for i, j in pairs:
            Xi, remap_i = self.submatrix(ranking['embeddings'], [k for k, x in enumerate(clusters) if x == i])
            l = [k for k, x in enumerate(clusters) if j < x < j + 2]
            Xj, remap_j = self.submatrix(ranking['embeddings'], l)
            similarity = self.compute_similarity(Xi, Xj)
            self.update_threshold(j) if self.decay else None
            for r in range(len(similarity)):
                most_similar = np.argmax(similarity[r])
                if similarity[r][most_similar] >= self.threshold:
                    res.append((topics[remap_i[r]], topics[remap_j[most_similar]], similarity[r][most_similar]))
                    if remap_i[r] in skipped:
                        remap_i.pop(remap_i[r])
                else:
                    skipped.add(remap_i[r])
                    # res.append((topics[remap_i[r]], -1, 0))

        for r in skipped:
            res.append((topics[r], -1, 0))

        return res, ranking


class SemanticClusterLinking(ClusterLinking):
    """
    Creates a taxonomy by first clustering the terms by theirs semantic, and then linking the terms in each cluster
    based on the order
    """

    def __init__(self, embedding: AbstractEmbeddingModel, embedding_key: str, clustering: AbstractClustering,
                 clustering_key: str = 'mean', threshold: float = 0.0, **kwargs):
        super().__init__(embedding, embedding_key, clustering, clustering_key, threshold)
        self.order_linker = OrderLinking(embedding, embedding_key)

    def create_taxonomy(self, ranking: DataFrame) -> Tuple[List[Tuple[str, str, int]], DataFrame]:
        _, clusters = self.cluster(ranking, order=False)
        res = []
        for c in set(clusters):
            sub_ranking = ranking[ranking['cluster'] == c]
            sub_taxo, _ = self.order_linker.create_taxonomy(sub_ranking)
            res.extend(sub_taxo)

        return res, ranking


class WikidataLinking(AbstractLinking):
    def __init__(self, embedding: AbstractEmbeddingModel, embedding_key, **kwargs):
        super().__init__(embedding, embedding_key, **kwargs)
        self.client = Client()

    def run(self, ranking: DataFrame) -> Tuple[List[Tuple[str, str, int]], DataFrame]:
        ranking = ranking.sort_values('mean', ascending=False).reset_index(drop=True)
        res, ranking = self.create_taxonomy(ranking)

        return res, ranking

    def create_taxonomy(self, ranking: DataFrame) -> Tuple[List[Tuple[str, str, int]], DataFrame]:
        q_ids = ranking['q_id']
        res = []
        remap = ranking.set_index('q_id')['topic'].to_dict()
        for q in q_ids:
            entity = self.client.get(q, load=True)
            candidates = self.extract_candidates(entity)
            intersection = candidates.intersection(set(q_ids))
            for element in intersection:
                link = (remap[q], remap[element], 0) if intersection else (remap[q], -1, 0)
                res.append(link)

        return res, ranking

    @staticmethod
    def extract_candidates(entity):
        candidates = set()
        try:
            iof = entity.attributes['claims']['P31']
            if iof:
                for iof_entity in iof:
                    try:
                        iof_qid = iof_entity['mainsnak']['datavalue']['value']['id']
                        candidates.add(iof_qid)
                    except KeyError:
                        continue
        except KeyError:
            return candidates

        return candidates
