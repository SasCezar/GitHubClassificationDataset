import hydra
import pandas
from hydra.utils import instantiate
from omegaconf import DictConfig


def query_augmentation(topic):
    return [f'what is {topic}', f'definition {topic}', topic]


@hydra.main(config_path="conf", config_name="scraper")
def scrape(cfg: DictConfig):
    engine = instantiate(cfg.search_engine)
    ranking = pandas.read_csv(cfg.ranking_path)
    for qid, topic in zip(ranking['q_id'], ranking['topic']):
        with open(f'{qid}.txt', 'wt') as outf:
            queries = query_augmentation(topic)
            res = set()

            for query in queries:
                results = engine.search(query)
                links = results.links()
                res.update(links)

            for link in res:
                outf.write(f'{link}\n')


if __name__ == '__main__':
    scrape()
