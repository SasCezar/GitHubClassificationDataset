library(ggplot2)


df <- read.csv("/home/sasce/PycharmProjects/GitHubClassificationDataset/data/all_topics_freq.csv", stringsAsFactors = FALSE)

p <- ggplot(df, aes(x = as.numeric(row.names(df)), y = freq)) + 
  geom_line(col = "#ff5a32",show.legend = TRUE) +
  #scale_x_continuous(trans = 'log10') + 
  scale_y_continuous(trans = 'log10')+
  annotation_logticks(sides="lb") + 
  geom_vline(xintercept = 3000, linetype="dashed", 
             color = "blue", size=1) +
  labs(x = 'N', y = "Frequency", color='Cluster') +
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
ggsave('topics_dist_log.bmp',
       width = 8, height = 5, device=cairo_pdf())




