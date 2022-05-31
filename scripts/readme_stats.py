import json

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="extract_readme")
def readme_stats(cfg: DictConfig):
    """
    Basic statistics regarding the size of the README content
    :param cfg:
    :return:
    """
    stats = {'null': 0, 'lengths': 0, 'size': 0}
    with open(cfg.out_path, 'rt') as inf:
        for line in inf:
            obj = json.loads(line)
            readme = obj['readme_text']
            if not readme:
                stats['null'] += 1
            else:
                stats['lengths'] += len(readme)

            stats['size'] += 1
    stats['avg_lengths'] = stats['lengths']/(stats['size'] - stats['null'])
    print(stats)


if __name__ == '__main__':
    readme_stats()
