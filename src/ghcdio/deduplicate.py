import json

import pandas as pd


def duplicate2json(path):

    df = pd.read_csv(path)

    df.drop(['map'], axis=1, inplace=True)
    df2 = df.copy(deep=True)
    df2.set_index(['topic'], drop=True, inplace=True)
    confidence = df2['confidence_score'].to_dict()
    df.drop(['confidence_score'], axis=1, inplace=True)

    res = df.groupby('Cluster ID')['topic'].apply(list).to_dict()
    with open('/src/run/csv_example_output_list.json', 'wt') as outf:
        for k in res:
            topics = sorted([(x, confidence[x]) for x in res[k]], key=lambda x: x[1], reverse=False)
            r = {'cluster': k, 'topics': topics}
            kv = json.dumps(r) + '\n'
            outf.write(kv)