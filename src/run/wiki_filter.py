import csv

import pandas

from processing.wiki import WikiRedirectNormalizer, WikiReconciler


def filter_topics():
    topics = pandas.read_csv('filtered.csv')
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

    reconciler = WikiReconciler()
    top = 10
    reconciled = reconciler.filter(topics, top=top)
    reconciled.to_csv(f'reconciled_{top}.txt', index=False)

if __name__ == '__main__':
    filter_topics()
