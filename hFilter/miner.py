import os
from pathlib import Path
from time import sleep, time

from github import Github, RateLimitExceededException, UnknownObjectException
from ratelimit import limits, sleep_and_retry

TIME_INTERVAL = 3600
CALLS = 1000
NL = "\n"
GITHUB_KEY = "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"
REPO_LIST_FILE = "repo_list_part_70000.txt"
README_FOLDER = "/Users/juri/Downloads/repologue_dataset"
OUTPUT_CSV = "topics_map.csv"


@sleep_and_retry
@limits(calls=CALLS, period=TIME_INTERVAL)
def mine_github_repo(g, repo_name):
    with open(OUTPUT_CSV, 'a', encoding='utf-8', errors='ignore') as results:
        try:
            file_name = repo_name.replace('/', ',') + '.txt'
            projectRepo = g.get_repo(repo_name)
            topics_repo = projectRepo.get_topics()
            with open(os.path.join(README_FOLDER, file_name), 'w', encoding='utf-8', errors='ignore') as res:
                try:
                    res.write(str(projectRepo.get_readme().decoded_content))
                    sleep(1)
                except:
                    print("error decoding readme")
        except RateLimitExceededException:
            print("rate limit exceeded")
            return
        except UnknownObjectException:
            print("Repo not found")


def get_textual_content(repo):
    with open(os.path.join(README_FOLDER, f'{repo.full_name.replace("/", ",").strip()}.txt'), 'w', encoding='utf-8',
              errors='ignore') as res:
        files = repo.get_contents("./")
        res.write(repo.description + '\n')
        while len(files) > 0:
            file_content = files.pop(0)
            if file_content.type == 'dir':
                files.extend(repo.get_contents(file_content.path))
            else:
                res.write(file_content.name + '\n')
                if file_content.name == 'README.md' or file_content.name == 'README' or file_content.name == 'readme.txt':
                    print('it is a readme ' + file_content.name)
                    # sleep(1)
                    res.write(str(file_content.decoded_content))


def main():
    topics = []
    g = Github(GITHUB_KEY, per_page=100)
    Path('./results').mkdir(parents=True, exist_ok=True)
    list_analyzed = os.listdir(README_FOLDER)
    with open(REPO_LIST_FILE, "r", encoding='utf-8', errors='ignore') as reader:
        for k, row1 in enumerate(reader):
            # owner, repo = row.replace("\n","").split("/")
            repo_name = row1.replace("\n", "")
            file_name = repo_name.replace('/', ',') + '.txt'
            if not file_name in list_analyzed:
                mine_github_repo(g, repo_name)


if __name__ == '__main__':
    start = time()
    main()
    end = time()
    print("seconds " + str(end - start))
