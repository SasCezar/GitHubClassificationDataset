import csv

import pandas
import wikipedia
from tqdm import tqdm
from wikipedia import DisambiguationError, PageError


def load_topics(path):
    df = pandas.read_csv(path)
    topics = df.iloc[:, 1]

    return list(set(topics))


def check_topic_online(topics):
    with open('is_app_domain.txt', 'wt') as outf:
        writer = csv.writer(outf)
        for topic in tqdm(topics):
            wiki_title = topic.replace('-', '_')

            try:
                wikipedia.page(wiki_title, auto_suggest=False)
                res = 1
            except (DisambiguationError, PageError):
                res = 0
            finally:
                writer.writerow([topic, res])


def check_topic_dump(topics, wiki_titles):
    count = 0
    with open('is_app_domain.txt', 'wt') as outf:
        writer = csv.writer(outf)
        for topic in tqdm(topics):
            wiki_title = topic.replace('-', '_').lower()

            res = wiki_title in wiki_titles
            if res:
                count += 1
            writer.writerow([topic, int(res)])

    print(count)
    return


def load_wiki(wiki_path):
    titles = set()
    with open(wiki_path, 'rt') as outf:
        for line in outf:
            title = line.split('\t')[-1].strip()
            titles.add(title.lower())

    return titles


def get_app_domain():
    file_path = '/home/sasce/PycharmProjects/GitHubClassificationDataset/hFilter/rules/split_dash_topics.csv'
    wiki_path = '/home/sasce/Downloads/enwiki-20211020-all-titles'
    topics = load_topics(file_path)
    wiki_titles = load_wiki(wiki_path)
    check_topic_dump(topics, wiki_titles)


if __name__ == '__main__':
    get_app_domain()
