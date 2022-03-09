import json
from collections import Counter

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="stats")
def count_freq(cfg: DictConfig):
    counter = Counter()
    with open(cfg.filepath, 'rt') as inf:
        for line in inf:
            proj_dict = json.loads(line)
            prog_lang = proj_dict['language']
            counter[prog_lang] += 1

    print(counter.most_common())
    print(sum(counter.values()))

if __name__ == '__main__':
    count_freq()