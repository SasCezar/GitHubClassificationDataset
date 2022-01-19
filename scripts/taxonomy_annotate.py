import glob
import os

import hydra
import pandas
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="taxonomy_annotate")
def join_taxonomies(cfg: DictConfig):
    taxonomies_path = glob.glob(f'{cfg.taxonomies_path}/*.csv')
    taxonomies_files = []
    for x in taxonomies_path:
        head, tail = os.path.split(x)
        taxonomies_files.append((x, tail))
    topics = pandas.read_csv(cfg.topics_path).set_index('q_id')

    for path, file in taxonomies_files:
        tax = pandas.read_csv(path)[['q_id', 'parent']].set_index('q_id')

        topics = topics.join(tax, on='q_id')
        topics.rename(columns={'parent': file.replace('.csv', '').replace('RankingClusterLinking', 'RCL')}, inplace=True)

    topics.sort_values('mean', ascending=True, inplace=True)
    topics.drop(['mean', 'std', 'description'], axis=1, inplace=True)

    topics.to_csv(cfg.out_file)


if __name__ == '__main__':
    join_taxonomies()
