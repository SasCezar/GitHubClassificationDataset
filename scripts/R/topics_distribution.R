library(ggplot2)
library(dplyr)

df <- read.csv("/home/sasce/PycharmProjects/GitHubClassificationDataset/data/all_topics_freq.csv", stringsAsFactors = FALSE)

fancy_scientific <- function(l) {
  # turn in to character string in scientific notation
  l <- format(round(as.numeric(l), 1), nsmall=0, big.mark=",") 
  # quote the part before the exponent to keep all the digits
  # return this as an expression
  l
}


p <- ggplot(df, aes(x = as.numeric(row.names(df)), y = freq)) + 
  geom_line(col = "#ff5a32", show.legend = TRUE) +
  #scale_x_continuous(trans = 'log10') + 
  scale_y_continuous(trans = 'log10', labels = fancy_scientific) +
  scale_x_continuous(labels = scales::comma)+
  annotation_logticks(sides="lb") + 
  geom_vline(xintercept = 3000, linetype="dashed", 
             color = "blue", size=1) +
  labs(x = 'N', y = "Frequency", color='Cluster') +
  theme(legend.title = element_blank(),
        #legend.position = c(0.15, 0.9),
        legend.position="top",
        legend.text = element_text(size=17),
        #        legend.background = element_rect(fill=alpha('white', 0)),
        #        panel.grid.major.y = element_blank(),
        #        panel.grid.major.x = element_blank(),
        #        panel.grid.minor.x = element_blank(),
        #        panel.grid.minor.y = element_blank(),
        axis.text=element_text(size=17),
        axis.title=element_text(size=17),
        axis.text.x = element_text(angle = 45, hjust = 1))
ggsave('topics_dist_log.pdf',
       width = 8, height = 5, device=cairo_pdf())