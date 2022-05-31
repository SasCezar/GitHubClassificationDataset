import csv
from collections import Counter

import hydra
import pandas as pd
from omegaconf import DictConfig


def load_ties(ties_path):
    res = []
    with open(ties_path, 'rt') as inf:
        reader = csv.reader(inf)
        next(reader)
        for line in reader:
            res.append(line)

    return res


def map_ties(ties, topics_id):
    mapping = topics_id.set_index('id')['name'].to_dict()
    return [(mapping[int(x[0])], mapping[int(x[1])]) for x in ties]


def measure_ties(mapped, clusters):
    cluster_map = clusters.set_index('topic')['cluster'].to_dict()
    tot = len(mapped)
    same = 0
    count = Counter()
    for x, y in mapped:
        try:
            if cluster_map[x] == cluster_map[y] or abs(cluster_map[x] - cluster_map[y]):
                same += 1
                count[cluster_map[x]] += 1
                count[cluster_map[y]] += 1
            else:
                print(x, y, cluster_map[x], cluster_map[y])
        except:
            continue

    return same, tot, same / tot, count


@hydra.main(config_path="conf", config_name="tie_analysis")
def tie_eval(cfg: DictConfig):
    """
    Measure the number of ties in the annotated pairs
    :param cfg:
    :return:
    """
    ties = load_ties(cfg.ties_file)
    topics_id = pd.read_csv(cfg.db_topics)
    clusters = pd.read_csv(cfg.clusters)
    mapped = map_ties(ties, topics_id)
    score = measure_ties(mapped, clusters)
    print(score[:3])
    print(score[3].most_common())


if __name__ == '__main__':
    tie_eval()
