import csv
from abc import ABC, abstractmethod
from os.path import join
from typing import List, Tuple

from networkx import write_graphml, DiGraph
from pandas import DataFrame, Series


class AbstractTaxonomyExporter(ABC):
    def __init__(self, out_path=".", out_name="taxonomy"):
        self.out_path = out_path
        self.out_name = out_name

    @abstractmethod
    def export(self, nodes, elements: List[Tuple]) -> None:
        raise NotImplemented


class GraphMLExporter(AbstractTaxonomyExporter):
    def __init__(self, out_path=".", out_name="taxonomy"):
        super().__init__(out_path, out_name)
        self.graph = DiGraph()

    def export(self, nodes, edges: List[Tuple]) -> None:
        self.create_graph(nodes, edges)
        write_graphml(self.graph, join(self.out_path, f'{self.out_name}.graphml'))

    def create_graph(self, nodes: DataFrame, edges: List[Tuple]) -> None:
        nodes = self.make_nodes(nodes)
        edges = self.make_edges(edges)
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)

    def make_edges(self, edges):
        size = len(edges[0])
        edges = [(x, y, {'weight': w}) for x, y, w in edges] if size == 3 else edges

        return edges

    def make_nodes(self, nodes: DataFrame):
        nodes = nodes.to_dict('records')
        return [(x['topic'], x) for x in nodes]


class CSVExporter(AbstractTaxonomyExporter):
    def __init__(self, out_path=".", out_name="taxonomy"):
        super().__init__(out_path, out_name)

    def export(self, topics: DataFrame, elements: List[Tuple]) -> None:
        topic_q_id = topics.set_index('topic').to_dict()['q_id']

        with open(join(self.out_path, f'{self.out_name}.csv'), 'wt') as outf:
            writer = csv.writer(outf)
            writer.writerow(['q_id', 'child', 'parent', 'similarity'])
            for edge in elements:
                writer.writerow([topic_q_id[edge[0]], *edge])
