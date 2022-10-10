import json
from collections import Counter

import hydra
import numpy as np
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="dataset")
def datastats(cfg: DictConfig):
    """
    Computes some statistics on the content of the dataset
    :param cfg:
    :return:
    """
    lengths = []
    num_labels = Counter()
    avg_labels = []
    with open(cfg.out_path, "r") as inf:
        for line in inf:

            doc = json.loads(line)
            if len(doc["labels"]) == 0:
                continue
            readme = doc["readme_text"]
            lengths.append(len(readme.split(' ')))
            print(doc['labels'], doc['topics']) if len(doc['labels']) > 12 else None
            num_labels[len(doc['labels'])] += 1
            avg_labels.append(len(doc['labels']))

    print('Total', len(lengths))
    print('Average', np.mean(lengths))
    print('STD', np.std(lengths))
    print('Min', np.min(lengths))
    print('Max', np.max(lengths))
    print(num_labels.most_common())
    print('Average labels', np.mean(avg_labels))
    print('STD labels', np.std(avg_labels))
    print('Annotations', np.sum(avg_labels))


if __name__ == '__main__':
    datastats()
