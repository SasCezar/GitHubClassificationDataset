library(ggplot2)
library(dplyr)
library(scales)
library(ggrepel)
library(reshape2)

df <- read.csv("/home/sasce/PycharmProjects/GitHubClassificationDataset/data/results/all_categories/results_avg.csv",
               col.names = c("Success Rate","Precision","Recall","F1"),
               stringsAsFactors = FALSE)
df$Success.Rate <- df$Success.Rate * 100
df <- melt(df, variable.name = "metric", 
           value.name = "value")

ggplot(df, aes(x = metric, y = value, color=metric)) +
  geom_violin(fill = "grey80", show.legend = FALSE) +
  #  labs(x = 'Similarity', y = "Intersection") +
  theme_linedraw() +
  theme(legend.title = element_blank(),
        legend.text = element_text(size=14),
        legend.background = element_rect(fill=alpha('white', 0)),
        panel.grid.major.y = element_blank(), #element_line(color = "grey80"),
        panel.grid.major.x = element_blank(), #element_line(color = "grey80"),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        axis.text=element_text(size=14),
        axis.title=element_text(size=14),
        axis.text.x = element_text(angle = 45, hjust = 1))

#ggsave("",
#       width = 8, height = 6, device=cairo_pdf())
