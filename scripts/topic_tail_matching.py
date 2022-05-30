import csv
import json

import hydra
from omegaconf import DictConfig
from thefuzz import fuzz


def load_label_mapping(label_path):
    res = {}
    rev = {}
    with open(label_path, 'r') as inf:
        for line in inf:
            label_map = json.loads(line)
            terms = label_map['GitHub Topic'] + [label_map['Wikidata Title']] + label_map['Wikidata Aliases']
            for topic in terms:
                res[topic] = label_map['Wikidata ID']
                rev[label_map['Wikidata ID']] = label_map['Wikidata Title']

    return res, rev


@hydra.main(config_path="conf", config_name="topic_tail_matching")
def match_tail(cfg: DictConfig):
    term_qid, qid_term = load_label_mapping(cfg.topics_path)

    res = {}
    with open(cfg.candidates_path, 'r') as inf:
        for line in inf:
            obj = json.loads(line)
            term = obj['term']
            candidates = obj['candidates']
            for candidate in candidates:
                cqid = candidate['id']
                if cqid in qid_term and fuzz.partial_ratio(term, qid_term[cqid]) > 75 and qid_term[cqid].lower() != term.lower().replace('-', ' '):
                    res[term] = qid_term[cqid]
                    continue
    print(len(res))
    with open(cfg.out_mapping, 'wt') as outf:
        writer = csv.writer(outf)
        writer.writerow(['original', 'remapped', 'qid'])
        for k in res:
            writer.writerow([k, res[k], term_qid[res[k]]])


if __name__ == '__main__':
    match_tail()
