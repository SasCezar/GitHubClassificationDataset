import csv
import os
from collections import defaultdict
from glob import glob


def parse_name(file):
    file = os.path.basename(file)
    file = file.replace('.txt', '')
    file = file.split(',')
    assert len(file) == 2, file
    return file[0], file[1]


if __name__ == '__main__':
    folder_path = '../data/preprocessed'

    labels = glob(os.path.join(folder_path, "*", ""))
    annotations = defaultdict(list)
    for label in labels:
        files = glob(os.path.join(label, "*.txt"))
        for file in files:
            author, project = parse_name(file)

            annotations[(author, project)].append(label.replace(folder_path, '').replace('/', ''))

    with open('../data/annotations.txt', 'wt', encoding='utf8') as outf:
        writer = csv.writer(outf)

        for author, project in annotations:
            row = [author, project]
            row.extend(annotations[(author, project)])
            writer.writerow(row)
