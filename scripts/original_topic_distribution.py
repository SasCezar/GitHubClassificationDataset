import csv
from collections import Counter


def count_freq(file_path):
    count = Counter()
    with open(file_path, 'rt') as inf:
        reader = csv.reader(inf)
        for line in reader:
            topics = line[4:]

            count.update(topics)

    return count


if __name__ == '__main__':
    file_path = '../hFilter/dataset_j/INPUT_CSV/topics_raw.csv'

    count = count_freq(file_path)

    print(count.most_common(2000)[1000:])
    with open('../data/all_topics_freq.csv', 'wt') as outf:
        writer = csv.writer(outf)
        for word, c in count.most_common():
            writer.writerow([word, c])

