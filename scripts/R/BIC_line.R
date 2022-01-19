library(ggplot2)
library(dplyr)
library(scales)
library(ggrepel)


df <- read.csv("/home/sasce/PycharmProjects/GitHubClassificationDataset/notebooks/BIC_scores_full.csv", stringsAsFactors = FALSE)
ggplot(df, aes(x = x, y = y)) +
  geom_line(col = "#ff5a32",show.legend = TRUE) +
  geom_point() + 
  guides(colour = guide_legend(nrow = 1))+
  #geom_text(aes(label=label),hjust=0, vjust=0) +
  labs(x = 'K', y = "BIC score") +
  #geom_text_repel() +

  theme(legend.title = element_blank(),
        #legend.position = c(0.15, 0.9),
        legend.position="top",
        legend.text = element_text(size=14),
        axis.text=element_text(size=14),
        axis.title=element_text(size=14),
        axis.text.x = element_text(angle = 45, hjust = 1))+ 
  scale_x_continuous(breaks= pretty_breaks())  +
ggsave('BIC_scores_full.pdf',
       width = 8, height = 5, device=cairo_pdf())
