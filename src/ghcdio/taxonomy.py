import csv
from abc import ABC, abstractmethod
from os.path import join
from typing import List, Tuple

from networkx import write_graphml, DiGraph
from pandas import DataFrame


class AbstractTaxonomyExporter(ABC):
    def __init__(self, out_path="."):
        self.out_path = out_path

    @abstractmethod
    def export(self, nodes, elements: List[Tuple]) -> None:
        raise NotImplemented


class GraphMLExporter(AbstractTaxonomyExporter):
    def __init__(self, out_path):
        super().__init__(out_path)
        self.graph = DiGraph()

    def export(self, nodes, edges: List[Tuple]) -> None:
        self.create_graph(nodes, edges)
        write_graphml(self.graph, join(self.out_path, 'taxonomy.graphml'))

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
    def __init__(self, out_path="."):
        super().__init__(out_path)

    def export(self, _, elements: List[Tuple]) -> None:
        with open('taxonomy.csv', 'wt') as outf:
            writer = csv.writer(outf)
            for edge in elements:
                writer.writerow(list(edge))
