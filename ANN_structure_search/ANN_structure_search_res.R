setwd("/home/abbas/projects/180710_EEG_DeepL/ANN_structure_search/")
library(stringr)

struct_data = read.csv('Dog_1_ANN_structure_search.log',sep = ';', header = FALSE)
colnames(struct_data) <- c('subject','channels_used','neurons','auc_cv_mean','auc_cv_std')

struct_data$n_hidden_layers <- str_count(struct_data$neurons, ",")
struct_data$index = rownames(struct_data)
library(ggplot2)


ggplot(struct_data)



ggplot(struct_data[struct_data$channels_used==3,],aes(x= as.factor(n_hidden_layers), y = auc_cv_mean))+
    theme_bw()+
   geom_violin()+coord_cartesian(ylim=c(0.5,1))+ geom_jitter(height = 0, width = 0.1,size=0.6,aes(color=index==2287))+
  xlab('number of hidden layers') + ylab('area under ROC curve')+
  theme(text = element_text(size=18))+
  ggtitle('3 channels used')+
  scale_color_manual(values=c("black", "#D55E00"))+
  theme(legend.position="none")+
  geom_text(data=struct_data[struct_data$index== 2287&struct_data$channels_used==3,],aes(label=neurons),hjust=0.5, vjust=-1,size=5,color='#D55E00')
ggsave('structure_search_dog1_3ch.png',dpi=500,width = 8,height=6)
#indx of top points 2287
#, 2034,2322,1635


ggplot(struct_data[struct_data$channels_used==2,],aes(x= as.factor(n_hidden_layers), y = auc_cv_mean))+
  theme_bw()+
  geom_violin()+coord_cartesian(ylim=c(0.5,1))+ geom_jitter(height = 0, width = 0.1,size=0.6,aes(color=index==455))+
  xlab('number of hidden layers') + ylab('area under ROC curve')+
  theme(text = element_text(size=18))+
  ggtitle('2 channels used')+
  scale_color_manual(values=c("black", "#D55E00"))+
  theme(legend.position="none")+
  geom_text(data=struct_data[struct_data$index== 455&struct_data$channels_used==2,],aes(label=neurons),hjust=0.5, vjust=-1,size=5,color='#D55E00')
ggsave('structure_search_dog1_2ch.png',dpi=500,width = 8,height=6)

  #geom_text(aes(label=index),hjust=0, vjust=0,size=5,color='#D55E00')
#455


ggplot(struct_data[struct_data$channels_used==7,],aes(x= as.factor(n_hidden_layers), y = auc_cv_mean))+
  theme_bw()+
  geom_violin()+coord_cartesian(ylim=c(0.5,1))+ geom_jitter(height = 0, width = 0.1,size=0.6,aes(color=index==3567))+
  #geom_violin()+coord_cartesian(ylim=c(0.5,1))+ geom_jitter(height = 0, width = 0.1,size=0.6)+
  xlab('number of hidden layers') + ylab('area under ROC curve')+
  theme(text = element_text(size=18))+
  ggtitle('7 channels used')+
  scale_color_manual(values=c("black", "#D55E00"))+
  theme(legend.position="none")+
  #geom_text(aes(label=index),hjust=0, vjust=0,size=5,color='#D55E00')
  geom_text(data=struct_data[struct_data$index== 3567&struct_data$channels_used==7,],aes(label=neurons),hjust=0.5, vjust=-1,size=5,color='#D55E00')
ggsave('structure_search_dog1_7ch.png',dpi=500,width = 8,height=6)



