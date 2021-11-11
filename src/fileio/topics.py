import csv
import json
from ast import literal_eval
from os import listdir
from os.path import join, isfile
from typing import Dict


def load_reconciled(filepath: bytes, freq: Dict[str, int]) -> Dict:
    """
    Loads the candidates from the Reconciler output for a specific term
    :param filepath:
    :param freq:
    :return:
    """
    reconciled = {'candidates': []}

    with open(filepath, 'rt') as inf:
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


def reconciled2json(freq, reconciled_path, data, out_path):
    """
    Exports the reconciled files to a JSON file where each line represents a term
    :param freq:
    :param reconciled_path:
    :param data:
    :param out_path:
    :return:
    """
    reconciled_path = join(reconciled_path, data)
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
    with open(join(out_path, f'{data}.json'), 'wt') as outf:
        for x in res:
            try:
                reconciled_json = json.dumps(x, ensure_ascii=False) + '\n'
            except:
                print(x)

            outf.write(reconciled_json)

    return res


def export2csv(disambiguated,  out_path, policy, data='gittopic', n=3000):
    """
    Exports the disambiguated topics to csv
    :param disambiguated:
    :param policy:
    :param data:
    :param n:
    :return:
    """
    write = []
    for dis in disambiguated:
        if not dis:
            continue
        topic = dis['term']
        candidate_id = -1
        candidate_name = 'None'
        ctypes = []
        if dis['best']:
            candidate_id = dis['best']['id']
            candidate_name = dis['best']['name']
            ctypes = [dis['best']['types'][x] for x in dis['best']['types']]

        freq = dis['frequency']
        row = [topic, freq, '', candidate_id, candidate_name]
        row.extend(ctypes)
        write.append(row)

    sorted_dis = sorted(write, key=lambda x: x[1], reverse=True)

    with open(join(out_path, f'disambiguated-{data}_{policy}.csv'), 'wt') as outf:
        writer = csv.writer(outf)
        for row in sorted_dis[:n]:
            writer.writerow(row)


def types2csv(types, out):
    """
    Exports the Wikidata types to CSV
    :param types:
    :param out:
    :return:
    """
    with open(join(out, 'wiki_types.csv'), 'wt') as outf:
        writer = csv.writer(outf)
        for (id, term), count in types.most_common():
            writer.writerow([id, term, count])
