rm(list = ls())
gc()

library(data.table)
library(ggplot2)

# library(plyr)
n_obs <- 81
n_moves <- 20
data_dir <- '/Users/habib.adam/dev/bibzzzz/pedestrian_blob/sim_data/'

blob_data_path <- paste0(data_dir, 'path_data_expl_0.2_move_limit_', n_moves, '_coord_range_4.csv')

blob_data <- read.csv(blob_data_path)
blob_data <- data.table(blob_data)

# simID_filter <- '4e462af5-2007-4387-a568-bc7b6798442c'
# sim_setID_filter <- 1
sim_set <- unique(blob_data$simID)[sample(1:length(unique(blob_data$simID)), length(unique(blob_data$simID)), replace = FALSE)][1:n_obs]
plot_data <- subset(blob_data, simID%in%sim_set)


plot <- ggplot(data = subset(plot_data, decision_state='post'), aes(x=sim_blob_x, y=sim_blob_y, group=interaction(simID), colour=factor(sim_result))) +
# plot <- ggplot(data = subset(plot_data, decision_state='post'), aes(x=sim_blob_x, y=sim_blob_y, group=interaction(simID), colour=sim_n_moves)) +
  geom_path(alpha=1) +
  geom_point(aes(x=sim_target_x, y=sim_target_y), shape=13, size=3, colour="yellow") +
  geom_point() +
  geom_point(data = subset(plot_data, decision_stage=='pre' & sim_moveID==0), aes(x=sim_blob_x, y=sim_blob_y), shape=1, colour="green") +

  xlim(c(-4, 4)) +
  ylim(c(-4, 4)) +
  facet_wrap(~simID) +
  theme(
        # axis.line=element_blank(),
        axis.text.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks=element_blank(),
        axis.title.x=element_blank(),
        axis.title.y=element_blank(),
        # legend.position="none",
        panel.background=element_rect(fill = 'black'),
        panel.border=element_blank()
        # panel.grid.major=element_blank(),
        # panel.grid.minor=element_blank(),
        # plot.background=element_blank()
        )

plot

subset(plot_data, simID=="03b8f353")
