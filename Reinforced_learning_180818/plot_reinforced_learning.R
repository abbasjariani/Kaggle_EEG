setwd('/media/abbas/Extension_3TB/180710_EEG_DeepL/Reinforced_learning_180818')


reinf_data = read.csv('reinforced_learning.log', sep = ';', header = FALSE)
colnames(reinf_data ) <- c('primary_subj','secondary_subj','roc_auc_no_sec_train','roc_auc_sec_train_cv_mean','roc_auc_sec_train_cv_std')


reinf_data$roc_auc_sec_noreinf_cv_mean = NA
#getting these values from the results of greedy search
reinf_data[reinf_data$secondary_subj=='Dog_1','roc_auc_sec_noreinf_cv_mean'] <- 0.8351041666666668
reinf_data[reinf_data$secondary_subj=='Dog_2','roc_auc_sec_noreinf_cv_mean'] <- 0.9308333333333334
reinf_data[reinf_data$secondary_subj=='Dog_3','roc_auc_sec_noreinf_cv_mean'] <- 0.9019841269841269
reinf_data[reinf_data$secondary_subj=='Dog_4','roc_auc_sec_noreinf_cv_mean'] <- 0.8338991500490355
reinf_data[reinf_data$secondary_subj=='Dog_5','roc_auc_sec_noreinf_cv_mean'] <- 0.9181481481481482
reinf_data[reinf_data$secondary_subj=='Patient_1','roc_auc_sec_noreinf_cv_mean'] <- 0.99
reinf_data[reinf_data$secondary_subj=='Patient_2','roc_auc_sec_noreinf_cv_mean'] <- 0.8284722222222222

library(ggplot2)

cur_sub = 'Dog_1'
ggplot(reinf_data[reinf_data$secondary_subj==cur_sub,])+theme_bw()+
  geom_point(aes(x=primary_subj,y=roc_auc_sec_train_cv_mean))+
  geom_hline(yintercept= unique(reinf_data[reinf_data$secondary_subj==cur_sub,'roc_auc_sec_noreinf_cv_mean']),color='#0072B2')+
  annotate("text", x = 4, y =  unique(reinf_data[reinf_data$secondary_subj==cur_sub,'roc_auc_sec_noreinf_cv_mean'])+0.01, label = "no pre-training",color='#0072B2',size=6)+
  ylab('area under ROC curve') + xlab('subject used to pre-train the model')+
  ggtitle(cur_sub)+
  theme(text = element_text(size=18))
ggsave('pretrain_effect_dog1.png',dpi=500,width=8,height = 5)
  
cur_sub = 'Dog_2'
ggplot(reinf_data[reinf_data$secondary_subj==cur_sub,])+theme_bw()+
  geom_point(aes(x=primary_subj,y=roc_auc_sec_train_cv_mean))+
  geom_hline(yintercept= unique(reinf_data[reinf_data$secondary_subj==cur_sub,'roc_auc_sec_noreinf_cv_mean']),color='#0072B2')+
  annotate("text", x = 4, y =  unique(reinf_data[reinf_data$secondary_subj==cur_sub,'roc_auc_sec_noreinf_cv_mean'])+0.002, label = "no pre-training",color='#0072B2',size=6)+
  ylab('area under ROC curve') + xlab('subject used to pre-train the model')+
  ggtitle(cur_sub)+
  theme(text = element_text(size=18))
ggsave('pretrain_effect_dog2.png',dpi=500,width=8,height = 5)

cur_sub = 'Dog_3'
ggplot(reinf_data[reinf_data$secondary_subj==cur_sub,])+theme_bw()+
  geom_point(aes(x=primary_subj,y=roc_auc_sec_train_cv_mean))+
  geom_hline(yintercept= unique(reinf_data[reinf_data$secondary_subj==cur_sub,'roc_auc_sec_noreinf_cv_mean']),color='#0072B2')+
  annotate("text", x = 4, y =  unique(reinf_data[reinf_data$secondary_subj==cur_sub,'roc_auc_sec_noreinf_cv_mean'])+0.001, label = "no pre-training",color='#0072B2',size=6)+
  ylab('area under ROC curve') + xlab('subject used to pre-train the model')+
  ggtitle(cur_sub)+
  theme(text = element_text(size=18))
ggsave('pretrain_effect_dog3.png',dpi=500,width=8,height = 5)


cur_sub = 'Dog_4'
ggplot(reinf_data[reinf_data$secondary_subj==cur_sub,])+theme_bw()+
  geom_point(aes(x=primary_subj,y=roc_auc_sec_train_cv_mean))+
  geom_hline(yintercept= unique(reinf_data[reinf_data$secondary_subj==cur_sub,'roc_auc_sec_noreinf_cv_mean']),color='#0072B2')+
  annotate("text", x = 4, y =  unique(reinf_data[reinf_data$secondary_subj==cur_sub,'roc_auc_sec_noreinf_cv_mean'])+0.01, label = "no pre-training",color='#0072B2',size=6)+
  ylab('area under ROC curve') + xlab('subject used to pre-train the model')+
  ggtitle(cur_sub)+
  theme(text = element_text(size=18))
ggsave('pretrain_effect_dog4.png',dpi=500,width=8,height = 5)



cur_sub = 'Patient_1'
ggplot(reinf_data[reinf_data$secondary_subj==cur_sub,])+theme_bw()+
  geom_point(aes(x=primary_subj,y=roc_auc_sec_train_cv_mean))+
  geom_hline(yintercept= unique(reinf_data[reinf_data$secondary_subj==cur_sub,'roc_auc_sec_noreinf_cv_mean']),color='#0072B2')+
  annotate("text", x = 4, y =  unique(reinf_data[reinf_data$secondary_subj==cur_sub,'roc_auc_sec_noreinf_cv_mean'])+0.01, label = "no pre-training",color='#0072B2',size=6)+
  ylab('area under ROC curve') + xlab('subject used to pre-train the model')+
  ggtitle(cur_sub)+
  theme(text = element_text(size=18))
ggsave('pretrain_effect_patient1.png',dpi=500,width=8,height = 5)


cur_sub = 'Patient_2'
ggplot(reinf_data[reinf_data$secondary_subj==cur_sub,])+theme_bw()+
  geom_point(aes(x=primary_subj,y=roc_auc_sec_train_cv_mean))+
  geom_hline(yintercept= unique(reinf_data[reinf_data$secondary_subj==cur_sub,'roc_auc_sec_noreinf_cv_mean']),color='#0072B2')+
  annotate("text", x = 4, y =  unique(reinf_data[reinf_data$secondary_subj==cur_sub,'roc_auc_sec_noreinf_cv_mean'])+0.01, label = "no pre-training",color='#0072B2',size=6)+
  ylab('area under ROC curve') + xlab('subject used to pre-train the model')+
  ggtitle(cur_sub)+
  theme(text = element_text(size=18))
ggsave('pretrain_effect_patient2.png',dpi=500,width=8,height = 5)