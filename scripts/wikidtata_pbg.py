import hydra
import pandas as pd
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="wikidata_pbg")
def extract_vector(cfg: DictConfig):
    embedding_path = cfg.embedding_path
    topic_path = cfg.topic_path
    out = cfg.out_path

    topic_q_id = pd.read_csv(topic_path)['q_id']
    url_q_id = {f"<http://www.wikidata.org/entity/{q_id}>" for q_id in topic_q_id}
    i = 0
    with open(embedding_path, 'rt') as inf, open(out, 'wt') as outf:
        for line in inf:
            if line.split(' ')[0] in url_q_id:
                outf.write(line + '\n')
                i += 1

    print('Total', len(url_q_id), 'Found', i)


if __name__ == '__main__':
    extract_vector()
