import csv
import json
from ast import literal_eval
from collections import Counter
from os import listdir
from os.path import isfile, join

import pandas
import pandas as pd


def load_skip_types():
    df = pandas.read_csv('annotated_wikitopics.csv')
    skip = {x for x, v in zip(df['id'], df['skip']) if v == 0}
    return skip


skip_types = load_skip_types()


def load_reconciled(file, freq):
    reconciled = {'candidates': []}

    with open(file, 'rt') as inf:
        reader = csv.reader(inf)
        next(reader)
        for i, line in enumerate(reader):
            reconciled['term'] = line[-1]
            reconciled['frequency'] = freq[line[-1]]
            if len(line) == 3:
                return reconciled
            candidate = {'id': line[0], 'match': eval(line[1]), 'name': line[2], 'score': float(line[3]), 'result_n': i}
            try:
                candidate_types = literal_eval(line[4])
            except:
                candidate_types = line[4]
            types = {}
            if isinstance(candidate_types, list):
                for c_type in candidate_types:
                    types[c_type['id']] = c_type['name']
            else:
                types[line[5]] = candidate_types

            candidate["types"] = types
            reconciled['candidates'].append(candidate)

    return reconciled


def reconciled2json(freq, data='reconciled-gittopic'):
    reconciled_path = f'/home/sasce/PycharmProjects/GitHubClassificationDataset/src/run/{data}/'
    reconciled_files = [join(reconciled_path, f) for f in listdir(reconciled_path) if isfile(join(reconciled_path, f))]
    res = []

    seen = set()
    for file in reconciled_files:
        reconciled = load_reconciled(file, freq)
        if reconciled['term'] in seen:
            continue
        seen.add(reconciled['term'])
        res.append(reconciled)

    res = sorted(res, key=lambda x: x['frequency'], reverse=True)
    with open(f'../../data/{data}.json', 'wt') as outf:
        for x in res:
            try:
                reconciled_json = json.dumps(x, ensure_ascii=False) + '\n'
            except:
                print(x)

            outf.write(reconciled_json)

    return res


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


def first_disambiguate(reconciled):
    disambiguated = []

    for rec in reconciled:
        rec['best'] = rec['candidates'][0] if rec['candidates'] else {}
        disambiguated.append(rec)

    return disambiguated


def is_clean(x):
    types = {k for k in x['types']}
    clean = bool(types.intersection(skip_types))
    return not clean


def first_clean_disambiguate(reconciled):
    disambiguated = []

    for rec in reconciled:
        candidates = [x for x in rec['candidates'] if is_clean(x)]
        rec['best'] = candidates[0] if candidates else {}
        disambiguated.append(rec)

    return disambiguated


def disambiguate(reconciled, policy='match'):
    disambiguated = None
    if policy == 'match':
        disambiguated = match_disambiguate(reconciled)

    if policy == 'first':
        disambiguated = first_disambiguate(reconciled)

    if policy == 'first-clean':
        disambiguated = first_clean_disambiguate(reconciled)

    return disambiguated


def get_types(reconciled, top=10):
    types = Counter()
    for rec in reconciled:
        for candidate in rec['candidates'][:top]:
            types.update([(k, v) for k, v in candidate['types'].items()])

    return types


def get_candidate_type(candidate):
    for tk in candidate['types']:
        c_type = candidate['types'][tk]
        if 'language' in c_type:
            return 'Programming Language'
        if 'free software' in c_type:
            return 'Software'
        if 'business' in c_type:
            return 'Business'
        if 'framework' in c_type or 'library' in c_type:
            return 'Library'
        if 'computing platform' in c_type:
            return 'Computing Platform'
        else:
            return 'UNK'
    return 'UNK'

def export2csv(disambiguated, policy, data='gittopic'):
    write = []
    for dis in disambiguated:
        if not dis:
            continue
        topic = dis['term']
        candidate_id = -1
        candidate_name = 'None'
        candidate_type = 'UNK'
        if dis['best']:
            candidate_id = dis['best']['id']
            candidate_name = dis['best']['name']
            candidate_type = get_candidate_type(dis['best'])
        freq = dis['frequency']
        write.append([topic, candidate_id, candidate_name, freq, candidate_type])

    sorted_dis = sorted(write, key=lambda x: x[3], reverse=True)

    with open(f'disambiguated-{data}_{policy}.csv', 'wt') as outf:
        writer = csv.writer(outf)
        for topic, candidate_id, candidate_name, freq, candidate_type in sorted_dis:
            writer.writerow([topic, candidate_id, candidate_name, freq, candidate_type])


def types2csv(types):
    with open('wiki_types.csv', 'wt') as outf:
        writer = csv.writer(outf)
        for (id, term), count in types.most_common():
            writer.writerow([id, term, count])


if __name__ == '__main__':
    res = 'reconciled-gittopic'
    data = 'github'
    freq = pd.read_csv('../../data/all_topics_freq.csv').set_index('topic')['freq'].to_dict()
    reconciled = reconciled2json(freq, res)
    types = get_types(reconciled, top=5)
    policy = 'first-clean'
    disambiguated = disambiguate(reconciled, policy=policy)
    print(types.most_common(100))
    export2csv(disambiguated, policy, data)
    types2csv(types)
