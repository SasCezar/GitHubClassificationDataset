from collections import Counter

import pandas as pd

from src.fileio.topics import reconciled2json, export2csv, types2csv
from src.processing.disambiguate import FirstCleanDisambiguate


def match_disambiguate(reconciled):
    disambiguated = []

    for rec in reconciled:
        rec['best'] = {}
        for candidate in rec['candidates']:
            if candidate['match']:
                best_match = candidate
                rec['best'] = best_match
                break

        disambiguated.append(rec)
    return disambiguated


def get_types(reconciled, top=10):
    types = Counter()
    for rec in reconciled:
        for candidate in rec['candidates'][:top]:
            types.update([(k, v) for k, v in candidate['types'].items()])

    return types


if __name__ == '__main__':
    res = 'reconciled-gittopic'
    data = 'github'
    freq = pd.read_csv('../../data/all_topics_freq.csv').set_index('topic')['freq'].to_dict()
    disambiguate = FirstCleanDisambiguate('annotated_wikitopics')
    reconciled = reconciled2json(freq, res)

    disambiguated = disambiguate.run(reconciled)
    export2csv(disambiguated, disambiguate.name, data)

    types = get_types(reconciled, top=5)
    types2csv(types)
