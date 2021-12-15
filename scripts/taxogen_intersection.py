import pandas


def intersect_our():
    topics_path = '/home/sasce/PycharmProjects/GitHubClassificationDataset/data/pairrank/mean_std_ranking.csv'
    taxogen_path = '/home/sasce/PycharmProjects/GitHubClassificationDataset/data/dblp/input/keywords.txt'

    topics = pandas.read_csv(topics_path)['topic']

    taxogen = pandas.read_csv(taxogen_path, header=None)[0].to_list()

    set_int = len(set(topics).intersection(set(taxogen)))

    same_int = 0

    partial_int = 0
    seen = {}
    for t in topics:
        for c in taxogen:
            if t.lower() == c.lower() or t.replace(' ', '_').lower() == c.lower():
                same_int += 1
                break
            if t.lower() in c.lower() or t.replace(' ', '_').lower() in c.lower():
                if t not in seen:
                    partial_int += 1
                    seen[t] = True
                print(t, c)


    print('set int', set_int)
    print('same int', same_int)
    print('partial int', partial_int)


def intersect_git():
    topics_path = '/home/sasce/PycharmProjects/GitHubClassificationDataset/data/all_topics_freq.csv'
    taxogen_path = '/home/sasce/PycharmProjects/GitHubClassificationDataset/data/dblp/input/keywords.txt'

    topics = pandas.read_csv(topics_path).fillna('')['topic']

    topics = [x.replace('-', '_') for x in topics]

    taxogen = pandas.read_csv(taxogen_path, header=None)[0].to_list()

    set_int = set(topics).intersection(set(taxogen))

    print(set_int)

    set_int= len(set_int)
    same_int = 0

    partial_int = 0
    seen = {}
    #for t in topics:
    #    for c in taxogen:
    #        if t.lower() == c.lower() or t.replace('-', '_').lower() == c.lower():
    #            same_int += 1
    #            break
            #if t.lower() in c.lower() or t.replace('-', '_').lower() in c.lower():
            #    if t not in seen:
            #        partial_int += 1
            #        seen[t] = True
            #    #print(t, c)

    print('set int', set_int)
    print('same int', same_int)
    print('partial int', partial_int)


if __name__ == '__main__':
    #intersect_our()
    intersect_git()