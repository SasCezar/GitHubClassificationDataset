import csv

import hydra
import numpy
import pandas
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.ghcdio.taxonomy2graph import AbstractTaxonomyExporter
from src.ml.clustering import AbstractClustering
from src.processing.linking import Linking


def export_taxonomy(res):
    with open('taxonomy.csv', 'wt') as outf:
        writer = csv.writer(outf)
        for line in res:
            writer.writerow(line)


@hydra.main(config_path="../conf", config_name="build_taxonomy")
def build_taxonomy(cfg: DictConfig):
    embedding = instantiate(cfg.embedding)
    print(cfg.linking._target_)
    clustering = instantiate(cfg.clustering) if cfg.clustering else None
    if 'ClusterLinking' in cfg.linking._target_:
        linking: Linking = instantiate(cfg.linking, embedding=embedding, clustering=clustering)
    else:
        linking: Linking = instantiate(cfg.linking, embedding=embedding)
    ranking = pandas.read_csv(cfg.ranking_path).fillna('')
    res, ranking = linking.run(ranking)
    print(res)
    export_taxonomy(res)
    exporter: AbstractTaxonomyExporter = instantiate(cfg.exporter)
    clustering: AbstractClustering = instantiate(cfg.clustering)
    arr = numpy.array(ranking['mean'].to_list())
    ranking['cluster'] = clustering.fit(arr)
    ranking.drop(['embeddings'], axis=1, inplace=True)
    exporter.export(ranking, res)


if __name__ == '__main__':
    build_taxonomy()
