import os
import pandas as pd
import shutil
import preprocessing

def serach_file(dir, filename):
    if os.path.isfile(os.path.join(dir,filename)):
        return os.path.join(dir, filename)
    # for dirpath, dirnames, filenames in os.walk(dir):
    #     for filename in [f for f in filenames if f==filename ]:
    #         return os.path.join(dirpath, filename)
def export(df, dir, out, featured = False):
    if featured:
        print("featured")
        topics_set = preprocessing.get_topic_frequency_map_topics_as_list(df)
        topics_set = {k: v for k, v in
                                     sorted(topics_set.items(), key=lambda item: item[1], reverse=True)}
    for index, row in df.iterrows():
        filename = f"{row['repo'].replace('/',',')}.txt"
        source = serach_file(dir,filename)
        if source == None:
            # print(filename)
            continue
        for topic in row['topics']:
            if(not featured or topic in topics_set):
                path = os.path.join(out,topic)
                if not os.path.isdir(path):
                    os.makedirs(path)
                shutil.copy(source, os.path.join(path,filename))
        # if index % 100 == 0 : print(index)
def remove_repository_without_readme_file(df, dir):
    for index, row in df.iterrows():
        filename = f"{row['repo'].replace('/',',')}.txt"
        source = serach_file(dir, filename)
        if source == None:
            df.drop(index, inplace=True)
    return df

