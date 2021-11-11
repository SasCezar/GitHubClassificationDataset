import pandas as pd
import os.path
def export(dataset, directory):

    projects_list = []
    for index, row in dataset.iterrows():
        #print(row['repo'], row['topics'])
        projects_list.append(row["repo"].replace("/","___"))
        dicth_filename = f'dicth_{row["repo"].replace("/","___")}'
        graph_filename = f'graph_{row["repo"].replace("/","___")}'
        with open(os.path.join(directory, dicth_filename),"w") as writer:
            writer.write(f'1\t{row["repo"].replace("/", "___")}\n')

            for i,topic in enumerate(row['topics']):
                writer.write(f'{i+2}\t#DEP#{topic}\n')
        with open(os.path.join(directory, graph_filename),"w") as writer:
            for i in range(2,len(row['topics'])+2):
                writer.write(f"1#{str(i)}\n")
    with open(os.path.join(directory,"projects.txt"), "w") as writer:
        for proj in projects_list:
            writer.write(f"{proj}\n")