from abc import abstractmethod, ABC
from collections import defaultdict
from itertools import product
from pprint import pprint
from typing import Iterable, List, Tuple, Optional, Dict

import numpy
import numpy as np
from numpy import ndarray, array
from pandas import DataFrame
from rdflib import Graph
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import fuzz
from wikidata.client import Client

from ml.clustering import AbstractClustering
from ml.embeddings import AbstractEmbeddingModel


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

    def run(self, ranking: DataFrame) -> Tuple[List[Dict], List[Tuple[str, str, int]], DataFrame]:
        ranking = ranking.sort_values('mean', ascending=False).reset_index(drop=True)
        if self.embedding:
            ranking['embeddings'] = self.get_embeddings(ranking[self.embedding_key])
        nodes, edges, ranking = self.create_taxonomy(ranking)

        return nodes, edges, ranking

    @abstractmethod
    def create_taxonomy(self, ranking: DataFrame) -> Tuple[List[Dict], List[Tuple[str, str, int]], DataFrame]:
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


class CSOLinking(AbstractLinking):
    def __init__(self, embedding: AbstractEmbeddingModel, embedding_key, **kwargs):
        super().__init__(embedding, embedding_key, **kwargs)
        self.rdf = Graph()
        self.rdf.parse("/home/sasce/PycharmProjects/GitHubClassificationDataset/data/cso/CSO.3.3.nt")
        self.wikidata = Client()

        self.wikidata_entity_query = """
            SELECT DISTINCT ?entity ?url
            WHERE {
                ?entity owl:sameAs ?url .
            FILTER CONTAINS(str(?url), "http://www.wikidata.org/entity/")
            }
        """

        self.label_entity_query = """
            SELECT DISTINCT ?entity ?label
            WHERE {?entity rdfs:label ?label .
        } """

        self.entity_query = """
            SELECT DISTINCT ?entity
            WHERE {?entity ?p ?o .
        } """

        # taxonomic lineage
        self.ancestors_query = """
            PREFIX csop: <http://cso.kmi.open.ac.uk/schema/cso#>
            PREFIX csoe: <https://cso.kmi.open.ac.uk/topics/>
            SELECT ?parent
            WHERE {{
                ?parent csop:superTopicOf <{topic}> .
            }}
        """

        self.aliases = None


    @staticmethod
    def extract_qid(url):
        return url.split('/')[-1]

    def create_taxonomy(self, ranking: DataFrame) -> Tuple[List[Dict], List[Tuple[str, str, int]], DataFrame]:
        topics = ranking.copy(deep=True).set_index('topic')['q_id'].to_dict()
        q_ids = ranking.set_index('q_id')['topic'].to_dict()

        gr_nodes = ranking.set_index('q_id').to_dict(orient='index')

        entities, reverse = self.get_cso_entities(topics, q_ids)
        # pprint(reverse)
        # pprint(self.aliases)

        print(len(entities))

        entities, reverse = self.disambiguate_entities(reverse, self.aliases)
        pprint(entities)

        print(len(entities))

        ancestors = self.get_entities_ancestors(entities)

        nodes = {}
        edges = set()

        added_qids = set()
        wikidata_entities = {self.normalize(str(x.entity)): self.extract_qid(x.url) for x in self.rdf.query(self.wikidata_entity_query)}
        print('WIKIDATA ENTITIES')
        pprint(wikidata_entities)
        for entity in ancestors:
            for (src, trg) in ancestors[entity]:
                edges.add((src, trg, 0))
                for x in (src, trg):
                    topic = x.split('/')[-1]
                    node = {'id': x, 'topic': topic, 'q_id': ''}

                    if x in entities:
                        node['q_id'] = entities[x]
                    else:
                        print('entity', x)
                        if x in wikidata_entities and wikidata_entities[x] not in entities:
                            print('ADDING QID')
                            node['q_id'] = wikidata_entities[x]

                    if node['q_id'] in gr_nodes:
                        node.update(gr_nodes[node['q_id']])

                    nodes[node['id']] = node
                    added_qids.add(node['q_id'])

        # for qid in gr_nodes:
        #     if qid in added_qids:
        #         continue
        #     node = gr_nodes[qid]
        #     node.update({'q_id': qid, 'id': qid})
        #     nodes[node['id']] = node

        print('nodes', len(nodes))
        print('egdes', len(edges))

        return list(nodes.values()), list(edges), ranking

    def get_cso_entities(self, topics: Dict, q_ids: Dict):
        wikidata_entities = self.rdf.query(self.wikidata_entity_query)
        label_entities = self.rdf.query(self.label_entity_query)
        entities = self.rdf.query(self.entity_query)
        res = {}

        aliases, self.aliases = self.get_aliases(q_ids)

        for r in label_entities:
            if str(r.label) in topics:
                norm = self.normalize(r.entity)
                res[norm] = topics[str(r.label)]

        for r in wikidata_entities:
            q = self.extract_qid(r.url)
            if q in q_ids:
                norm = self.normalize(r.entity)
                res[norm] = q

        for r in entities:
            clean_entity = str(r.entity).split('/')[-1].replace('_', ' ').replace('-', ' ').lower()

            if clean_entity in aliases:
                norm = self.normalize(r.entity)
                res[norm] = aliases[clean_entity]

        rev = defaultdict(list)
        for i in res:
            rev[res[i]].append(i)

        return res, rev

    def get_entities_ancestors(self, cso_entities):
        res = {}
        for entity in cso_entities:
            ancestors = self.get_ancestors(entity)
            res[entity] = ancestors

        return res

    def normalize(self, entity):
        query = f"""
                    PREFIX csop: <http://cso.kmi.open.ac.uk/schema/cso#>
                    PREFIX csoe: <https://cso.kmi.open.ac.uk/topics/>
                    SELECT ?normalized
                    WHERE {{
                        <{entity}> csop:preferentialEquivalent ?normalized .
                    }}
        """

        res = list(self.rdf.query(query))
        if res:
            assert len(res) == 1
            return str(res[0].normalized)

        return str(entity)

    def get_ancestors(self, entity):
        res = set()
        query = self.ancestors_query.format(topic=entity)
        q_res = list(self.rdf.query(query))
        for r in q_res:
            res.add((self.normalize(str(entity)), self.normalize(str(r.parent))))
            res.update(self.get_ancestors(r.parent))

        return res

    def get_aliases(self, q_ids):
        aliases = {}
        reverse = defaultdict(list)
        for q in q_ids:
            entity = self.wikidata.get(q, load=True)
            try:
                res = entity.attributes['aliases']['en']
                reverse[q] = [alias['value'].lower() for alias in res]
                reverse[q].append(entity.label['en'])
                assert len(reverse[q]) > 0, print(entity, res)
                for alias in res:
                    clean_entity = alias['value'].lower().replace('_', ' ').replace('-', ' ')
                    aliases[clean_entity] = q
            except:
                continue

        return aliases, reverse

    def disambiguate_entities(self, entities, aliases):
        reverse = {}
        for q_id in entities:
            candidates = entities[q_id]
            entity_aliases = aliases[q_id]
            if len(candidates) == 1 or len(entity_aliases) == 0:
                reverse[q_id] = candidates[0]
                continue

            similarities = []
            pairs = list(product(candidates, entity_aliases))
            print(candidates, entity_aliases, pairs)
            for candidate, alias in set(pairs):
                sim = fuzz.ratio(candidate.replace('https://cso.kmi.open.ac.uk/topics/', '').lower().replace('_', ' ').replace('-', ' '), alias)
                similarities.append((candidate, alias, sim))

            best = sorted(similarities, key=lambda x: x[2], reverse=True)
            print(best[0])
            reverse[q_id] = best[0][0]

        res = {reverse[k]: k for k in reverse}

        return res, reverse
