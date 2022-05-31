import json
from collections import Counter

import hydra
import numpy as np
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="data_integration")
def datastats(cfg: DictConfig):
    """
    Computes some statistics on the content of the dataset
    :param cfg:
    :return:
    """
    lengths = []
    num_labels = Counter()
    with open(cfg.out_path, "r") as inf:
        for line in inf:

            doc = json.loads(line)
            readme = doc["readme_text"]
            lengths.append(len(readme.split(' ')))
            print(doc['labels'], doc['topics']) if len(doc['labels']) > 12 else None
            num_labels[len(doc['labels'])] += 1

    print('Total', len(lengths))
    print('Average', np.mean(lengths))
    print('STD', np.std(lengths))
    print('Min', np.min(lengths))
    print('Max', np.max(lengths))
    print(num_labels.most_common())


if __name__ == '__main__':
    datastats()
