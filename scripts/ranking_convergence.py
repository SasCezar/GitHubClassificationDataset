import csv
from collections import defaultdict
from os import listdir
from os.path import isfile, join

import hydra
import numpy
import pandas as pd
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="convergence")
def measure_convergence(cfg: DictConfig):
    rankings = [join(cfg.rank_path, f) for f in listdir(cfg.rank_path) if isfile(join(cfg.rank_path, f))]
    rankings = sorted(rankings)

    matrix = []
    positions = defaultdict(list)
    for f in rankings:
        df = pd.read_csv(f)
        df = df.sort_values('mean', ascending=False).reset_index(drop=True)
        rank = df['topic'].to_dict()
        for t in rank:
            positions[rank[t]] = t

        matrix.append([positions[x] for x in positions])

    matrix = numpy.array(matrix).transpose()

    diff = numpy.diff(matrix)
    avg = numpy.average(numpy.abs(diff), axis=0)
    with open('convergence.csv', 'wt') as outf:
        writer = csv.writer(outf)
        for n, v in zip(range(200, 5600, 200), avg):
            writer.writerow([n, v])


if __name__ == '__main__':
    measure_convergence()
