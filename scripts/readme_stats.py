import csv
import json
from collections import Counter
from pprint import pprint

import hydra
import pandas as pd
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="dataset")
def readme_stats(cfg: DictConfig):
    """
    Basic statistics regarding the size of the README content
    :param cfg:
    :return:
    """
    readme_size = []
    labels = Counter()
    labels_project = []
    with open(cfg.dataset_out, 'rt') as inf:
        for line in inf:
            obj = json.loads(line)
            readme_size.append(len(obj['readme_text'].split(' ')))
            labels.update(obj['labels'])
            labels_project.append(len(obj['labels']))

    df = pd.DataFrame.from_dict(labels, columns=['Count'], orient='index').reset_index().rename(
        columns={'index': 'Label'})
    df = df.sort_values('Count', ascending=False)
    df.to_csv('labels_count.csv', index=False)
    df = pd.DataFrame(readme_size, columns=["Size"])
    df.to_csv('readme_size.csv', index=False)

    df = pd.DataFrame(labels_project, columns=["Count"])
    df.to_csv('labels_project_count.csv', index=False)


if __name__ == '__main__':
    readme_stats()
