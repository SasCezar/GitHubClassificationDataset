import json

import hydra
from omegaconf import DictConfig
from langdetect import detect


@hydra.main(config_path="conf", config_name="dataset")
def dataset_export(cfg: DictConfig):
    """
    Exports a subset of the data for the classification task (Script for final dataset)
    :param cfg:
    :return:
    """
    diff = 0
    keys = ['full_name', 'readme', 'readme_text', 'topics', 'labels', 'levels', 'description', 'language']
    with open(cfg.out_path, "r") as inf, \
            open(cfg.dataset_out, 'w') as outf:
        for line in inf:
            obj = json.loads(line)
            try:
                lang = detect(obj['readme_text'])
            except:
                print(obj['readme_text'])
                continue
            if lang != 'en':
                diff += 1
            if obj['labels'] and lang == 'en':
                item = {key: obj[key] for key in keys}
                outf.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(diff)


if __name__ == '__main__':
    dataset_export()
