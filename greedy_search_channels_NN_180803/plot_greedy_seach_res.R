setwd("/media/abbas/Extension_3TB/180710_EEG_DeepL/greedy_search_channels_NN_180803/")

data_log_dir = "/media/abbas/Extension_3TB/180710_EEG_DeepL/greedy_search_channels_NN_180803/logs/"

data_path_1 = "/media/abbas/Extension_3TB/180710_EEG_DeepL/greedy_search_channels_NN_180803/logs/Dog_3_greedy_NN_channel_search.log"
data_path_2 = "/media/abbas/Extension_3TB/180710_EEG_DeepL/greedy_search_channels_NN_180803/logs/Dog_4_greedy_NN_channel_search.log"
data_path_3 = "/media/abbas/Extension_3TB/180710_EEG_DeepL/greedy_search_channels_NN_180803/logs/Dog_5_greedy_NN_channel_search.log"
data_path_4 = "/media/abbas/Extension_3TB/180710_EEG_DeepL/greedy_search_channels_NN_180803/logs/greedy_NN_channel_search_Dog1_Dog2.log"
data_path_5 = "/media/abbas/Extension_3TB/180710_EEG_DeepL/greedy_search_channels_NN_180803/logs/Patient_1_greedy_NN_channel_search.log"
data_path_6 = "/media/abbas/Extension_3TB/180710_EEG_DeepL/greedy_search_channels_NN_180803/logs/Patient_2_greedy_NN_channel_search.log"


data_1 <- read.csv(data_path_1, sep =  ';')
data_2 <- read.csv(data_path_2, sep =  ';')
data_3 <- read.csv(data_path_3, sep =  ';')
data_4 <- read.csv(data_path_4, sep =  ';')
data_5 <- read.csv(data_path_5, sep =  ';')
data_6 <- read.csv(data_path_6, sep =  ';')

data_all <- rbind(data_1,data_2,data_3,data_4,data_5,data_6)
library(stringr)

data_all$n_channels <- str_count(data_all$channels, ",") + 1

library(ggplot2)

ggplot(data_all,aes(x=n_channels,y=auc_cross_val_mean,color=subject))+
  geom_line()+ theme_bw()+geom_point()+
  geom_errorbar(aes(ymin=auc_cross_val_mean-auc_cross_val_std, ymax=auc_cross_val_mean+auc_cross_val_std, colour=subject), width=.1) +
  coord_cartesian(ylim=c(0.5,1))+
  scale_color_brewer(palette="Dark2")
  
