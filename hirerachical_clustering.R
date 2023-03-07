#install.packages("wordcloud")
library(wordcloud)
#install.packages("tm")
library(tm)
# ONCE: install.packages("Snowball")
## NOTE Snowball is not yet available for R v 3.5.x
## So I cannot use it  - yet...
##library("Snowball")
##set working directory
#install.packages("slam")
library(slam)
#install.packages("quanteda")
library(quanteda)
## ONCE: install.packages("quanteda")
## Note - this includes SnowballC
library(SnowballC)
library(arules)
##ONCE: install.packages('proxy')
library(proxy)


df <- read.csv("data/cleaned_data/df_tweets_cleaned.csv")

cleanCourpus <- df$tweets_cleaned

MyDTM <- read.csv("data/cleaned_data/df_tweets_cv.csv")
MyDTM <- MyDTM[,-(1:2)]

tweets_m <- (as.matrix(MyDTM))


nrow(tweets_m)

tweets_m <- scale(tweets_m)
CosineSim <- tweets_m / sqrt(rowSums(tweets_m * tweets_m))
CosineSim <- CosineSim %*% t(CosineSim)

#Convert to distance metric

D_Cos_Sim <- as.dist(1-CosineSim)

HClust_Ward_CosSim_SmallCorp2 <- hclust(D_Cos_Sim, method="ward.D2")

plot(HClust_Ward_CosSim_SmallCorp2, cex=.7, hang=-11,main = "Cosine Sim")
rect.hclust(HClust_Ward_CosSim_SmallCorp2, k=4,border = 2:6)
abline(h = 3, col = 'red')

suppressPackageStartupMessages(library(dendextend))
avg_dend_obj <- as.dendrogram(HClust_Ward_CosSim_SmallCorp2)
avg_col_dend <- color_branches(avg_dend_obj, h = 3)
plot(avg_col_dend)

png("dendro.png",width=1600,height=800)
plot(HClust_Ward_CosSim_SmallCorp2, cex = 0.8, hang = -1)
dev.off()


