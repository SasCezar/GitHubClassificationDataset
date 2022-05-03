from typing import List

import hydra
import numpy
import pandas
from hydra.utils import instantiate
from omegaconf import DictConfig

from ghcdio.taxonomy import AbstractTaxonomyExporter
from ml.clustering import AbstractClustering
from processing.linking import AbstractLinking



@hydra.main(config_path="../conf", config_name="build_taxonomy")
def build_taxonomy(cfg: DictConfig):
    # init_seeds()
    #embedding = instantiate(cfg.embedding)
    embedding = None
    clustering = instantiate(cfg.clustering) if cfg.clustering else None
    if 'ClusterLinking' in cfg.linking._target_:
        linking: AbstractLinking = instantiate(cfg.linking, embedding=embedding, clustering=clustering)
    else:
        linking: AbstractLinking = instantiate(cfg.linking, embedding=embedding)
    ranking = pandas.read_csv(cfg.ranking_path).fillna('')
    ranking['topic_desc'] = ranking['topic'] + ' ' + ranking['description']
    nodes, edges, ranking = linking.run(ranking)
    print(edges)
    clustering: AbstractClustering = instantiate(cfg.clustering)
    arr = numpy.array(ranking['mean'].to_list())
    ranking['cluster'] = clustering.fit(arr)
    try:
        ranking.drop(['embeddings'], axis=1, inplace=True)
    except:
        pass
    exporters: List[AbstractTaxonomyExporter] = []

    for _, exporter_conf in cfg.exporter.items():
        if "_target_" in exporter_conf:
            exporters.append(instantiate(exporter_conf, out_name=cfg.out_name))

    for exporter in exporters:
        exporter.export(nodes, edges)


if __name__ == '__main__':
    build_taxonomy()
