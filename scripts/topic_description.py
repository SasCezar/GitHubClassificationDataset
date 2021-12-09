import pandas
import tqdm
from pandas import DataFrame


def get_descriptions(q_ids):
    from wikidata.client import Client

    client = Client()

    descriptions = []
    for q in tqdm.tqdm(q_ids):
        entity = client.get(q, load=True)
        if 'en' in entity.description:
            descriptions.append(entity.description['en'])
        else:
            descriptions.append('')

    return descriptions


def integration():
    df: DataFrame = pandas.read_csv(
        '/home/sasce/PycharmProjects/GitHubClassificationDataset/data/pairrank/mean_std_ranking.csv')
    descriptions = get_descriptions(df['q_id'].tolist())
    df['description'] = descriptions

    df.to_csv('/home/sasce/PycharmProjects/GitHubClassificationDataset/data/pairrank/mean_std_ranking.csv', index=False)


if __name__ == '__main__':
    integration()
