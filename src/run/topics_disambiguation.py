from collections import Counter

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.fileio.topics import reconciled2json, export2csv, types2csv


def get_types(reconciled, top=10):
    types = Counter()
    for rec in reconciled:
        for candidate in rec['candidates'][:top]:
            types.update([(k, v) for k, v in candidate['types'].items()])

    return types


@hydra.main(config_path="../conf", config_name="disambiguation")
def topic_disambiguation(cfg: DictConfig):
    freq = pd.read_csv(cfg.freq_path).set_index('topic')['freq'].to_dict()

    reconciled = reconciled2json(freq, cfg.reconciled_folder_path, cfg.reconciled_type, cfg.out_path)
    types = get_types(reconciled, top=5)
    types2csv(types, cfg.out_path)

    disambiguator = instantiate(cfg.disambiguation)
    disambiguated = disambiguator.run(reconciled)
    export2csv(disambiguated, cfg.out_path, disambiguator.name, cfg.data)


if __name__ == '__main__':
    topic_disambiguation()
