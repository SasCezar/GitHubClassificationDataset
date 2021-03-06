import hydra
import pandas
from omegaconf import DictConfig

from processing.wiki import WikiReconciler


@hydra.main(config_path="../conf", config_name="reconcile")
def reconcile_topics(cfg: DictConfig):
    file_path = cfg.file_path
    topics = pandas.read_csv(file_path)
    topics = topics['topic'].tolist()
    reconciler = WikiReconciler(cfg.reconcile_out)
    reconciler.run(topics, top=10)


if __name__ == '__main__':
    reconcile_topics()
