import json

import hydra
from bs4 import BeautifulSoup
from omegaconf import DictConfig


def get_readme(html: str):
    soup = BeautifulSoup(html, 'html.parser')
    readme = soup.findAll("div", {"data-target": 'readme-toc.content'})
    try:
        readme = readme[0]
    except:
        readme = ''
    return readme


@hydra.main(config_path="conf", config_name="extract_readme")
def extract_readme(cfg: DictConfig):
    """
    Parses the HTML document and extracts the README HTML section and the text from the section
    :param cfg:
    :return:
    """
    with open(cfg.in_path, "r") as inf, open(cfg.out_path, "w") as out:
        for line in inf:
            doc = json.loads(line)
            try:
                html = doc['HTML']
                readme = get_readme(html)
                doc['readme'] = str(readme)
                doc['readme_text'] = readme.text

                out.write(json.dumps(doc, ensure_ascii=False) + '\n')
            except:
                doc['readme'] = ''
                doc['readme_text'] = ''
                out.write(json.dumps(doc, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    extract_readme()
