library(ggplot2)
library(dplyr)
library(scales)
library(ggrepel)


df <- read.csv("/home/sasce/PycharmProjects/GitHubClassificationDataset/notebooks/scatter.csv", stringsAsFactors = FALSE)
df$cluster <- as.factor(df$cluster)

ggplot(df, aes(x = pos, y = mean, color=cluster)) +
  geom_point(show.legend = TRUE, size = 1.5) + 
  guides(colour = guide_legend(nrow = 1, override.aes = list(shape = 15, size=5)))+
  #geom_text(data=subset(df, (pos %% 35) == 1 & pos < 280), aes(pos,mean,label=topic), position=position_jitter(width=df$x,height=df$y)) +
  labs(x = 'Ranking Position', y = "Mean", color='Cluster') +
  #geom_text_repel() +
  theme(legend.title = element_blank(),
        #legend.position = c(0.15, 0.9),
        legend.position="top",
        legend.text = element_text(size=14),
#        legend.background = element_rect(fill=alpha('white', 0)),
#        panel.grid.major.y = element_blank(),
#        panel.grid.major.x = element_blank(),
#        panel.grid.minor.x = element_blank(),
#        panel.grid.minor.y = element_blank(),
        axis.text=element_text(size=14),
        axis.title=element_text(size=14),
        axis.text.x = element_text(angle = 45, hjust = 1)) +  
  scale_color_brewer(palette="Paired") +
ggsave('scatter_ranking.pdf',
       width = 8, height = 5, device=cairo_pdf())
