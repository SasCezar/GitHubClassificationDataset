import json

import hydra
import pandas

from omegaconf import DictConfig
from tqdm import tqdm
from wikidata.client import Client


def get_wiki_info(ids, lang='en'):
    titles = []
    descriptions = []
    client = Client()
    aliases = []

    def get_aliases(entity, lang='en'):
        aliases = entity.data['aliases']
        res = [x['value'] for x in aliases[lang]] if aliases and lang in aliases else []
        return res

    for q_id in tqdm(ids):
        entity = client.get(q_id, load=True)
        label = entity.data['labels'][lang]['value']
        titles.append(label)
        description = entity.description.texts[lang] if lang in entity.description.texts else ''
        descriptions.append(description)
        al = get_aliases(entity, lang)
        al.append(label)
        aliases.append(al)

    return titles, descriptions, aliases


@hydra.main(config_path="../conf", config_name="extract")
def extract_annotated(cfg: DictConfig):
    df = pandas.read_csv(cfg.annotated_file)
    df = df[df['Sum'] > 1]
    disambiguated = df.groupby('Wikidata ID').agg({
        'Frequency': 'sum',
        'GitHub Topic': lambda x: list(x)
    }).reset_index()

    wiki_titles, wiki_description, wiki_aliases = get_wiki_info(disambiguated['Wikidata ID'].tolist())

    disambiguated['Wikidata Title'] = wiki_titles
    disambiguated['Wikidata Description'] = wiki_description
    disambiguated['Wikidata Aliases'] = wiki_aliases

    disambiguated.to_csv(cfg.out_file, index=False)
    with open(cfg.out_file.replace('.csv', '.json'), 'wt') as outf:
        for obj in disambiguated.to_dict(orient='records'):
            row = json.dumps(obj, ensure_ascii=False)
            outf.write(row + '\n')


if __name__ == '__main__':
    extract_annotated()
