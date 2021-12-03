import hydra
import pandas
from omegaconf import DictConfig

from src.processing.linking import Linking


@hydra.main(config_path="../conf", config_name="build_taxonomy")
def build_taxonomy(cfg: DictConfig):
    embedding = hydra.utils.instantiate(cfg.embedding)
    clustering = None
    #clustering = hydra.utils.instantiate(cfg.clustering) if cfg.clustering else None
    if clustering:
        linking: Linking = hydra.utils.instantiate(cfg.linking, embedding=embedding, clustering=clustering)
    else:
        linking: Linking = hydra.utils.instantiate(cfg.linking, embedding=embedding)
    ranking = pandas.read_csv(cfg.ranking_path)
    res = linking.run(ranking)
    print(res)


if __name__ == '__main__':
    build_taxonomy()
