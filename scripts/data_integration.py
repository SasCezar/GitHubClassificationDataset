import json
from pprint import pprint

import hydra
import pandas as pd
from bs4 import BeautifulSoup
from omegaconf import DictConfig


def get_readme(html: str):
    soup = BeautifulSoup(html, 'html.parser')
    readme = soup.findAll("div", {"data-target": 'readme-toc.content'})
    try:
        readme = readme[0]
    except:
        readme = ''
    return readme


def load_label_mapping(label_path):
    res = {}
    with open(label_path, 'r') as inf:
        for line in inf:
            label_map = json.loads(line)
            for topic in label_map['GitHub Topic']:
                res[topic] = label_map['Wikidata Title']

    return res


def load_label_levels(label_levels_path):
    df = pd.read_csv(label_levels_path)
    return df.set_index('topic')['cluster'].to_dict()


def load_tail_map(tail_map_path):
    df = pd.read_csv(tail_map_path)
    return df.set_index('original')['remapped'].to_dict()


@hydra.main(config_path="conf", config_name="data_integration")
def extract_readme(cfg: DictConfig):
    """
    Adds the information from GitRanking to the RAW dataset of GitHub projects with the READMEs files
    :param cfg:
    :return:
    """
    skipped = 0
    total = 0
    label_mapping = load_label_mapping(cfg.label_path)
    label_level = load_label_levels(cfg.label_levels_path)
    tail_remap = load_tail_map(cfg.tail_map_path)
    pprint(tail_remap)
    pprint(label_mapping)
    empty = 0
    unlabeled = 0

    with open(cfg.in_path, "r") as inf, open(cfg.out_path, "w") as out:
        for line in inf:
            total += 1
            doc = json.loads(line)
            try:
                html = doc['HTML']
                readme = get_readme(html)
                doc['readme'] = str(readme).strip()
                if readme.text.strip() == '':
                    empty += 1
                doc['readme_text'] = readme.text.strip()

                labels = {label_mapping[x] for x in doc['topics'] if x in label_mapping}
                tail_labels = {tail_remap[x] for x in doc['topics'] if x in tail_remap}
                labels.update(tail_labels)
                labels = list(labels)
                levels = [label_level[x] for x in labels]
                if len(labels) == 0:
                    unlabeled += 1
                doc['labels'] = labels
                doc['levels'] = levels
                out.write(json.dumps(doc, ensure_ascii=False) + '\n')
            except Exception as e:
                print(e)
                print('Skipped')
                skipped += 1

    print('Total', total)
    print('Skipped', skipped, skipped / total)
    print('Empty', empty, empty / total)
    print('Unlabeled', unlabeled, unlabeled / total)


if __name__ == '__main__':
    extract_readme()
