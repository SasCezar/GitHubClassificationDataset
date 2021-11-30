import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas
from omegaconf import DictConfig


@hydra.main(config_path="../conf", config_name="taxonomy_plot")
def plot_ranking(cfg: DictConfig):
    df = pandas.read_csv(cfg.ranking_file)
    df = df.sort_values('mean', ascending=False)
    i = 300-20
    n = 300
    topics = df['topic'][i:n]
    mean = df['mean']
    std = df['std'][i:n]
    bars = topics
    height = std
    y_pos = np.arange(len(bars))

    # Create horizontal bars
    plt.barh(y_pos, height)

    # Create names on the x-axis
    plt.yticks(y_pos, bars)
    plt.tight_layout()
    # Show graphic
    plt.show()


if __name__ == '__main__':
    plot_ranking()
