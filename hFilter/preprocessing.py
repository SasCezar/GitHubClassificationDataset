import re

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus import words as w
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

REWRITING_RULE_FOLDER = 'rules'

##Only the first time

import nltk
nltk.download('words')

stops = stopwords.words("english")
words = w.words()
wordswn = wn.words()
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def topics_string_to_list(df):
    def make_topics(topics):
        result = set()
        for t in topics.split(','):
            if t.strip() == '':
                continue
            result.add(t)
        return (list(result))

    df.topics = df.topics.apply(make_topics)
    return df


def apply_csv(file_name, df):
    edit_list = {}
    with open(file_name) as file:
        for line in file.read().strip('\n').split('\n'):
            items = line.strip(',').split(',')
            edit_list[items[1].lower()] = list(map(lambda x: x.lower(), items[2:]))
    all_t = set()

    def make_topics(topics):
        result = []
        for t in topics.split(','):
            if t.strip() == '':
                continue
            if t in edit_list:
                all_t.add(edit_list[t][0])
                if edit_list[t][0] == '-1':
                    continue
                elif edit_list[t][0] == '-2':
                    result.append(t)
                    continue
                result.extend(edit_list[t])
            else:
                result.append(t)
        return ",".join(list(set(result)))

    df.topics = df.topics.apply(make_topics)
    df = df[df.topics != '']
    return df


def aggregate_topics(file_name, df):
    edit_list = []
    with open(file_name) as file:
        for line in file.read().strip('\n').split('\n'):
            items = line.strip(',').split(',')
            # edit_list.append(([ps.stem(x) for x in items[2:]],ps.stem(items[1])))
            edit_list.append(([x.lower().strip() for x in items[2:]], items[1].lower().strip()))
    all_t = set()

    def make_topics(topics):
        result = []
        deleted = []
        # topics = [ps.stem(x) for x in topics.split(',')]
        topics = [x for x in topics.split(',')]
        for abr in edit_list:
            if set(abr[0]) & set(topics) == set(abr[0]):
                all_t.add(abr[1])
                deleted.extend(list(abr[0]))
                result.append(abr[1])
        for del_topics in set(deleted):
            topics.remove(del_topics)
        result.extend(topics)
        return ",".join(list(set(result)))

    df.topics = df.topics.apply(make_topics)

    return df


def aggregate_two_topics(file_name, df):
    edit_list = {}
    with open(file_name) as file:
        for line in file.read().strip('\n').split('\n'):
            items = line.rstrip(',').split(',')
            # edit_list.append(([ps.stem(x) for x in items[2:]],ps.stem(items[1])))
            edit_list[items[2].lower().strip()] = items[1].lower().strip()
    all_t = set()

    def make_topics(topics):
        result = []
        topics = [x for x in topics.split(',')]
        for topic in topics:
            if topic in edit_list:
                all_t.add(topic)
                result.append(edit_list[topic])
            else:
                result.append(topic)
        return ",".join(list(set(result)))

    df.topics = df.topics.apply(make_topics)
    return df


def make_col(file_name, df, col, edit_list=None):
    if not edit_list:
        edit_list = []
        with open(file_name) as file:
            for line in file.read().strip('\n').split('\n'):
                item = line.strip(',').split(',')
                edit_list.append(item[0].lower().strip())

    def make_topics(topics):
        result = []
        topics = [x for x in topics.split(',')]
        for topic in topics:
            if topic in edit_list:
                result.append(topic)
        return ",".join(list(set(result)))

    df[col] = df.topics.apply(make_topics)
    return df


def remove_topics_contains_digit(file_name, df):
    _ = list(df.topics.map(lambda x: x.split(',')))
    _ = [i for s in _ for i in s]
    topics_list = set(_)
    topics_freq = {}

    def add_to_dict(x):
        if x in topics_freq:
            topics_freq[x] += 1
        else:
            topics_freq[x] = 1

    for x in _:
        add_to_dict(x)
    counter = 0
    with open(file_name, "w") as file:
        has_number = lambda x: any(c.isdigit() for c in x)
        for t in topics_list:
            if has_number(t):
                _ = re.sub(r'\d+', '', t).strip('-').strip().replace('--', '-')
                if _ == '':
                    file.write(f'{topics_freq[t]},{t},-1,\n')
                else:
                    file.write(f"{topics_freq[t]},{t},{_},\n")
                counter += 1
    return file_name, counter


def remove_topics_contains_version(file_name, df):
    _ = list(df.topics.map(lambda x: x.split(',')))
    _ = [i for s in _ for i in s]
    topics_list = set(_)

    counter = 0
    with open(file_name, "w") as file:
        for t in topics_list:
            if re.search("v[\d\.]+", t):
                _ = re.sub(r'v[\d\.]+', '', t).strip('-').strip()
                if _ == '':
                    file.write(f'{t},-1,\n')
                else:
                    file.write(f"{t},{_.replace('--', '-')},\n")
                counter += 1
    return file_name, counter


def remove_plural_topics(file_name, df):
    _ = list(df.topics.map(lambda x: x.split(',')))
    _ = [i for s in _ for i in s]
    topics_list = set(_)
    topics_freq = {}

    def add_to_dict(x):
        if x in topics_freq:
            topics_freq[x] += 1
        else:
            topics_freq[x] = 1

    for x in _:
        add_to_dict(x)
    counter = 0
    with open(file_name, "w") as file:
        for t in topics_list:
            if t.endswith('s') and t[:-1] in topics_list:
                file.write(f"{topics_freq[t]},{t},{t[:-1]},{topics_freq[t[:-1]]},\n")
                counter += 1
    return file_name, counter


def split_dash_topics(file_name, df):
    _ = list(df.topics.map(lambda x: x.split(',')))
    _ = [i for s in _ for i in s]
    topics_list = set(_)
    topics_freq = {}

    def add_to_dict(x):
        if x in topics_freq:
            topics_freq[x] += 1
        else:
            topics_freq[x] = 1

    for x in _:
        add_to_dict(x)
    counter = 0
    with open(file_name, "w") as file:
        for t in topics_list:
            if '-' in t:
                file.write(f"{topics_freq[t]},{t},{','.join(t.split('-'))},\n")
                counter += 1
    return file_name, counter


def remove_topics_contains_top_topics(file_name, df, threshold=200):
    _ = list(df.topics.map(lambda x: x.split(',')))
    _ = [i for s in _ for i in s]
    topics_list = set(_)
    topics_freq = {}

    def add_to_dict(x):
        if x in topics_freq:
            topics_freq[x] += 1
        else:
            topics_freq[x] = 1

    for x in _:
        add_to_dict(x)
    counter = 0
    topics_freq_list = list(topics_freq.items())
    topics_freq_list.sort(key=lambda x: x[1], reverse=True)
    with open(file_name, "w") as file:
        for top_topic, _ in topics_freq_list[:threshold]:
            for t, __ in topics_freq_list[threshold:]:
                if top_topic in t:
                    file.write(
                        f"{topics_freq[t]},{t},{top_topic},{t.replace(top_topic, '').replace('--', '-').strip('-')},\n")
                    counter += 1
    return file_name, counter


def remove_low_freq_topics(file_name, df, threshold=20):
    _ = list(df.topics.map(lambda x: x.split(',')))
    _ = [i for s in _ for i in s]
    topics_list = set(_)
    topics_freq = {}

    def add_to_dict(x):
        if x in topics_freq:
            topics_freq[x] += 1
        else:
            topics_freq[x] = 1

    for x in _:
        add_to_dict(x)
    counter = 0
    topics_freq_list = list(topics_freq.items())
    with open(file_name, "w") as file:
        for t, _ in topics_freq_list:
            if _ < threshold:
                file.write(f"{counter},{t},-1,\n")
                counter += 1
    return file_name, counter


def get_topic_frequency_map_topics_as_string(df):
    _ = list(df.topics.map(lambda x: x.split(',')))
    _ = [i for s in _ for i in s]
    topics_freq = {}

    def add_to_dict(x):
        if x in topics_freq:
            topics_freq[x] += 1
        else:
            topics_freq[x] = 1

    for x in _:
        add_to_dict(x)
    counter = 0
    topics_freq_list = list(topics_freq.items())

    return topics_freq


def get_topic_frequency_map_topics_as_list(df):
    _ = list(df.topics.map(lambda x: x))
    _ = [i for s in _ for i in s]
    topics_freq = {}

    def add_to_dict(x):
        if x in topics_freq:
            topics_freq[x] += 1
        else:
            topics_freq[x] = 1

    for x in _:
        add_to_dict(x)
    counter = 0
    topics_freq_list = list(topics_freq.items())

    return topics_freq


def remove_stopwords_topic(file_name, df):
    _ = list(df.topics.map(lambda x: x.split(',')))
    _ = [i for s in _ for i in s]
    topics_list = set(_)
    topics_freq = {}

    def add_to_dict(x):
        if x in topics_freq:
            topics_freq[x] += 1
        else:
            topics_freq[x] = 1

    for x in _:
        add_to_dict(x)
    counter = 0
    with open(file_name, "w") as file:
        for t in (set(_) & set(stops)):
            file.write(f'{topics_freq[t]},{t},-1,\n')
            counter += 1
    return file_name, counter


def remove_stemmed_topic(file_name, df):
    _ = list(df.topics.map(lambda x: x.split(',')))
    _ = [i for s in _ for i in s]
    topics_list = set(_)
    topics_freq = {}

    def add_to_dict(x):
        if x in topics_freq:
            topics_freq[x] += 1
        else:
            topics_freq[x] = 1

    for x in _:
        add_to_dict(x)
    counter = 0
    with open(file_name, "w") as file:
        for t in topics_list:
            st = ps.stem(t)
            if st in topics_freq and t != st:
                file.write(f'{topics_freq[t]},{t},{st},{topics_freq[st]},\n')
                counter += 1
    return file_name, counter


def remove_lemmatize_topic(file_name, df):
    _ = list(df.topics.map(lambda x: x.split(',')))
    _ = [i for s in _ for i in s]
    topics_list = set(_)
    topics_freq = {}

    def add_to_dict(x):
        if x in topics_freq:
            topics_freq[x] += 1
        else:
            topics_freq[x] = 1

    for x in _:
        add_to_dict(x)
    counter = 0
    with open(file_name, "w") as file:
        for t in topics_list:
            st = lemmatizer.lemmatize(t)
            if st != t:
                file.write(f'{topics_freq[t]},{t},{st},{topics_freq.get(st)},\n')
                counter += 1
    return file_name, counter


def preprocess(df, MIN_TOPIC_LENGHT=4):
    # Use a breakpoint in the code line below to debug your script.
    df['original_topics'] = df.topics
    df = df[df['stars'] >= 10]

    # print(f'INITIAL: {len(preprocessing.get_topic_frequency_map(df))}')
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/topics_contains_version.csv', df)
    # print(f'VERSION: {len(preprocessing.get_topic_frequency_map(df))}')
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/topics_contains_number.csv', df)
    # print(f'NUMBER: {len(preprocessing.get_topic_frequency_map(df))}')
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/split_dash_topics.csv', df)
    # print(f'SPLIT-DASH_ {len(preprocessing.get_topic_frequency_map(df))}')
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/contains_top_topics.csv', df)
    # print(f'CONTAINS TOP TOPICS: {len(preprocessing.get_topic_frequency_map(df))}')
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/remove_plural_topics.csv', df)
    # print(f'PLURAL TOPICS: {len(preprocessing.get_topic_frequency_map(df))}')
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/low_freq_topics_1.csv', df)
    # print(f'LOW FREQ: {len(preprocessing.get_topic_frequency_map(df))}')
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/remove_stopwords_topic.csv', df)
    # print(f'STOP WORD {len(preprocessing.get_topic_frequency_map(df))}')
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/remove_lemmatize_topic.csv', df)
    # print(f'LEMMATIZE TOPIC: {len(preprocessing.get_topic_frequency_map(df))}')
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/delete.csv', df)
    # print(f'DELETE?: {len(preprocessing.get_topic_frequency_map(df))}')
    df = aggregate_two_topics(f'{REWRITING_RULE_FOLDER}/replace.csv', df)
    # print(f'REPLACE: {len(preprocessing.get_topic_frequency_map(df))}')
    df = aggregate_topics(f'{REWRITING_RULE_FOLDER}/abbr.csv', df)
    # print(f'ABBR: {len(preprocessing.get_topic_frequency_map(df))}')
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/contractions.csv', df)
    # print(f'CONTRACTION: {len(preprocessing.get_topic_frequency_map(df))}')
    df = apply_csv(f'{REWRITING_RULE_FOLDER}/contains_selected_topics.csv', df)
    # print(f'SELECTED TOPIC: {len(preprocessing.get_topic_frequency_map(df))}')

    # df = preprocessing.topics_string_to_list(df)
    # clear_dataset = df[df['topics'].map(len) >= MIN_TOPIC_LENGHT]

    clear_dataset = df[df.apply(lambda x: len(x['topics'].split(",")) > MIN_TOPIC_LENGHT, axis=1)]

    return topics_string_to_list(clear_dataset)
    # preprocessing.remove_topics_contains_version(f'{REWRITING_RULE_FOLDER}/topics_contains_version.csv')
    # preprocessing.remove_topics_contains_digit(f'{REWRITING_RULE_FOLDER}/topics_contains_number.csv', df)
    # preprocessing.split_dash_topics(f'{REWRITING_RULE_FOLDER}/split_dash_topics.csv')
    # preprocessing.remove_topics_contains_top_topics(f'{REWRITING_RULE_FOLDER}/contains_top_topics.csv',df)
    # preprocessing.remove_plural_topics(f'{REWRITING_RULE_FOLDER}/remove_plural_topics.csv', df)
    # preprocessing.remove_low_freq_topics(f'{REWRITING_RULE_FOLDER}/low_freq_topics.csv', 50)
    # preprocessing.split_dash_topics(f'{REWRITING_RULE_FOLDER}/split_dash_topics.csv', df)
    # preprocessing.remove_stopwords_topic(f'{REWRITING_RULE_FOLDER}/remove_stopwords_topic.csv', df)
    # preprocessing.remove_stemmed_topic(f'{REWRITING_RULE_FOLDER}/remove_stemmed_topic.csv', df)
    # preprocessing.remove_lemmatize_topic(f'{REWRITING_RULE_FOLDER}/remove_lemmatize_topic.csv', df)
