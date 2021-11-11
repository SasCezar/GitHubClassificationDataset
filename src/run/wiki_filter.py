import hydra
import pandas
from omegaconf import DictConfig

from src.processing.wiki import WikiReconciler


@hydra.main(config_path="../conf", config_name="config")
def filter_topics(cfg: DictConfig):
    file_path = cfg.file_path
    topics = pandas.read_csv(file_path)
    topics = topics['topic'].tolist()
    # filterer = WikiRedirectNormalizer(
    #     '/home/sasce/PycharmProjects/GitHubClassificationDataset/data/wikipedia/wiki_mapping.csv')
    #
    # filtered = filterer.filter(topics)
    # skipped = sum([1 for x in filtered if x == -1])
    # with open('filtered.csv', 'wt') as outf:
    #     writer = csv.writer(outf)
    #     for topic, map in zip(topics, filtered):
    #         writer.writerow([topic, map])
    #
    # print(skipped)

    reconciler = WikiReconciler('/home/sasce/PycharmProjects/GitHubClassificationDataset/src/run/reconciled-gittopic/')
    top = 10
    reconciled = reconciler.filter(topics, top=top)
    # reconciled.to_csv(f'reconciled_{top}.txt', index=False)


if __name__ == '__main__':
    filter_topics()
