import glob
import json
from collections import defaultdict
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm


def load_readme(readme_dir):
    files = [Path(x) for x in glob.glob(readme_dir + '/**/*.txt', recursive=True)]
    readme = {}
    classes = defaultdict(list)

    for file in tqdm(files):
        category = file.parent.name
        project = file.name.replace(',', '/').replace('.txt', '').lower()
        with open(file, 'rt') as inf:
            text = inf.read()
        readme[project] = text
        classes[project].append(category)

    return readme, classes


@hydra.main(config_path="conf", config_name="readme_integration")
def integrate_readme(cfg: DictConfig):
    readme_dir = cfg.readme_dir
    projects_path = cfg.projects_path

    readmes, labels = load_readme(readme_dir)
    res = []
    count = 0
    with open(projects_path, 'r') as inf:
        for line in inf:
            try:
                obj = json.loads(line)
                project_name = obj['full_name'].lower()
                obj['repologue_labels'] = labels[project_name]
                obj['readme'] = readmes[project_name]
                res.append(obj)
                count += 1
            except:
                continue

    with open(cfg.output_path, 'w') as outf:
        for r in res:
            text = json.dumps(r, ensure_ascii=False)
            outf.write(text + '\n')

    print(count)


if __name__ == '__main__':
    integrate_readme()
