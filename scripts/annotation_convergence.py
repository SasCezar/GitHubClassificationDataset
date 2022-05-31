import os
from collections import defaultdict
from os import listdir
from os.path import isfile, join

import hydra
import numpy
import pandas
from omegaconf import DictConfig


def final_rank(file_path):
    df = pandas.read_csv(file_path)
    df = df.sort_values('mean', ascending=False).reset_index(drop=True)
    rank = {v: k for k, v in df['topic'].to_dict().items()}
    return rank


@hydra.main(config_path="conf", config_name="annot_convergence")
def measure_convergence(cfg: DictConfig):
    """
    Script for measuring the number of annotation required for convergence as defined in the paper.
    :param cfg:
    :return:
    """
    topics_folders = [(d, os.path.join(cfg.topics_path, d)) for d in os.listdir(cfg.topics_path) if
                      os.path.isdir(os.path.join(cfg.topics_path, d))]
    positions = defaultdict(list)
    final = final_rank(join(cfg.topics_path, 'all.csv'))
    diffs = {}
    for topic, path in topics_folders:
        files = [(join(path, f), f) for f in listdir(path) if isfile(join(path, f))]
        files = sorted(files, key=lambda x: int(x[1].replace('.csv', '')))
        for file_path, _ in files:
            df = pandas.read_csv(file_path)
            df = df.sort_values('mean', ascending=False).reset_index(drop=True)
            rank = {v: k for k, v in df['topic'].to_dict().items()}
            positions[topic].append(rank[topic])

    for t in positions:
        d = [abs(final[t] - p) for p in positions[t]]
        diffs[t] = d

    items = []
    for t in diffs:
        count = 0
        for i, x in enumerate(diffs[t]):
            if x <= 3:
                count += 1
            else:
                count = 0

            if count == 2:
                items.append(i)
                break
            if i == len(diffs[t]) - 1:
                items.append(i)
                break

    print(len(items), items)
    print(numpy.average(items), numpy.std(items))

    avg_ann = numpy.average([len(diffs[t]) for t in diffs])
    print('Avg Annot', avg_ann)


if __name__ == '__main__':
    measure_convergence()
