rm(list = ls())
gc()

library(data.table)
library(ggplot2)

# library(plyr)

n_obs <- 25
n_moves <- 'inf'
coord_range <- 4
expl_rate <- 0.2
data_dir <- '/Users/habib.adam/dev/bibzzzz/pedestrian_blob/sim_data/'

blob_data_path <- paste0(data_dir, '0.2_expl_rate_', n_moves, '_move_limit_', coord_range, '_coord_range_sim_data.csv')

blob_data <- read.csv(blob_data_path)
blob_data <- data.table(blob_data)

blob_data <- subset(blob_data, ((decision_stage=='pre'&n_moves==0)|decision_stage=='post'))

# simID_filter <- '4e462af5-2007-4387-a568-bc7b6798442c'
# sim_setID_filter <- 1
sim_set <- unique(blob_data$simID)[sample(1:length(unique(blob_data$simID)), length(unique(blob_data$simID)), replace = FALSE)][1:n_obs]
plot_data <- subset(blob_data, simID%in%sim_set)


plot <- ggplot(data = subset(plot_data), aes(x=blob_x, y=blob_y, group=interaction(simID), colour=factor(intel_version))) +
# plot <- ggplot(data = subset(plot_data, decision_stage='post'), aes(x=sim_blob_x, y=sim_blob_y, group=interaction(simID), colour=sim_n_moves)) +
  geom_path(alpha=1) +
  geom_point(aes(x=target_x, y=target_y), shape=13, size=3, colour="yellow") +
  geom_point() +
  geom_point(data = subset(plot_data, decision_stage=='pre' & n_moves==0), aes(x=blob_x, y=blob_y), shape=21, colour="green", fill="green") +

  xlim(c(-coord_range, coord_range)) +
  ylim(c(-coord_range, coord_range)) +
  facet_wrap(~simID*intel_version) +
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

summary(subset(blob_data, intel_version==0))
summary(subset(blob_data, intel_version==1))
summary(subset(blob_data, intel_version==2))
summary(subset(blob_data, intel_version==3))
summary(subset(blob_data, intel_version==4))

subset(plot_data, simID=="cad60f9d")
