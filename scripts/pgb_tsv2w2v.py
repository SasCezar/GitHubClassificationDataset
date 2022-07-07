import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="wikidata_pbg")
def reformat(cfg: DictConfig):
    line_count = str(sum(1 for _ in open('/home/sasce/PycharmProjects/GitHubClassificationDataset/data/git_topics_wikidata_embeddings.vec')))
    with open('/home/sasce/PycharmProjects/GitHubClassificationDataset/data/git_topics_wikidata_embeddings.vec', 'rt') as inp, \
            open('/home/sasce/PycharmProjects/GitHubClassificationDataset/data/github_pbg_w2v.vec', 'wt') as outp:
        line = next(inp).split('\t')
        dimensions = str(len(line) - 1)
        outp.write(' '.join([line_count, dimensions]) + '\n')
        inp.seek(0)
        for line in inp:
            line = line.replace('<http://www.wikidata.org/entity/', '').replace('>', '')
            words = line.strip().split()
            outp.write(' '.join(words) + '\n')


if __name__ == '__main__':
    reformat()
