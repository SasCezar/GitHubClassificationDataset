import csv
import re
from typing import Dict

from tqdm import tqdm

WIKITITLE_ITEM_RE = re.compile(r"\(\d+,\d+,'[^']+','\w*',\d,\d,\d.\d+,'\d+','\d+',\d+,\d+,'\w+',\w+\)")
REDIRECT_ITEM_RE = re.compile(r"\(\d+,\d+,'[^']+','\w*','[^']*'\)")


def load_pages(path: str) -> Dict:
    mapping = {}
    with open(path, "rt", encoding='ISO-8859–1') as inf:
        for line in tqdm(inf):
            matches = WIKITITLE_ITEM_RE.findall(line)
            for match in matches:
                wikibase_tuple = match[1:-1].split(",")
                mapping[wikibase_tuple[2][1:-1]] = wikibase_tuple[0]

    return mapping


def load_redirect(path: str) -> Dict:
    mapping = {}
    with open(path, "rt", encoding='ISO-8859–1') as inf:
        for line in tqdm(inf):
            matches = REDIRECT_ITEM_RE.findall(line)
            for match in matches:
                wikibase_tuple = match[1:-1].split(",")
                mapping[wikibase_tuple[0]] = wikibase_tuple[2][1:-1]

    return mapping


def remap(wikipages, redirects, out):
    with open(out, 'wt') as outf:
        writer = csv.writer(outf)
        for page in wikipages:
            mapped = page
            page_id = wikipages[page]
            if page_id in redirects:
                mapped = redirects[page_id]

            writer.writerow([page, mapped])

    return


def join_wiki():
    titles_path = '/home/sasce/PycharmProjects/GitHubClassificationDataset/data/wikipedia/enwiki-20211020-page.sql'
    redirect = '/home/sasce/PycharmProjects/GitHubClassificationDataset/data/wikipedia/enwiki-20211020-redirect.sql'
    out = 'mapping.csv'

    wikipages = load_pages(titles_path)
    redirects = load_redirect(redirect)

    remap(wikipages, redirects, out)


if __name__ == '__main__':
    join_wiki()
