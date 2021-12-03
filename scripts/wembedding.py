import logging
import re
from bz2 import BZ2File

import hydra
from gensim.models.word2vec import LineSentence
from hydra.utils import instantiate
from omegaconf import DictConfig


def trigram(cfg):
    pattern = re.compile(
        (r'^<http://www.wikidata.org/entity/(Q\d+)> '
         r'<http://www.wikidata.org/prop/direct/(P\d+)> '
         r'<http://www.wikidata.org/entity/(Q\d+)>'),
        flags=re.UNICODE)

    with open(cfg.trigram_path, 'w') as f:
        for line in BZ2File(cfg.dump_path):
            line = line.decode('utf-8')
            match = pattern.search(line)
            if match:
                f.write(" ".join(match.groups()) + '\n')


def train(cfg):
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO)

    sentences = LineSentence(cfg.trigram_path)

    w2v = instantiate(cfg.model, sentences=sentences)
    w2v.save(cfg.model_out)


@hydra.main(config_path="conf", config_name="wembedding")
def wikidata_embedding(cfg: DictConfig):
    trigram(cfg)
    train(cfg)


if __name__ == '__main__':
    wikidata_embedding()
