library(arules)

#library(streamR)
library(rjson)
#install.packages("tokenizers")
library(tokenizers)

library(tidyverse)
library(plyr)
library(dplyr)
library(ggplot2)
#install.packages("syuzhet")
## sentiment analysis
#library(syuzhet)
library(stringr)
library(arulesViz) ## load last

tweets_df <- read.csv("data/cleaned_data/df_tweets_cleaned.csv")

corpus <- tweets_df$tweets_cleaned

TransactionTweetsFile <- "tweets_trans.csv"
trans <- file(TransactionTweetsFile)
tokens <- tokenizers::tokenize_words(corpus[1], stopwords = stopwords::stopwords("en"), lowercase = TRUE, strip_punct = TRUE, strip_numeric = TRUE, simplify = TRUE)
cat(unlist(str_squish(tokens)), "\n", file = trans, sep = ",")
close(trans)

trans <- file(TransactionTweetsFile, open = "a")
for(i in 2:nrow(tweets_df)){
  tokens <- tokenizers::tokenize_words(tweets_df$tweets_cleaned[i], stopwords = stopwords::stopwords("en"), lowercase = TRUE, strip_punct = TRUE, strip_numeric = TRUE, simplify = TRUE)
  cat(unlist(str_squish(tokens)), "\n", file = trans, sep = ",")
}
close(trans)

TweetTrans <- read.transactions(TransactionTweetsFile,
                                rm.duplicates = FALSE,
                                format = "basket",
                                sep = ",")
inspect(TweetTrans)
sample_trans <- sample(TweetTrans, 50)
summary(sample_trans)

TweetDf <- read.transactions(TransactionTweetsFile, header = FALSE, sep = ",")
inspect(TweetDf)

TweetTrans_rules <- arules::apriori(TweetDf,
                                    parameter = list(support = 0.0001, confidence = 0.0001, minlen = 2, maxlen = 6))

inspect(TweetTrans_rules[1:30])

SortedRules_sup <- sort(TweetTrans_rules, by = "support", decreasing = TRUE)
inspect(SortedRules_sup[1:20])

SortedRules_conf <- sort(TweetTrans_rules, by = "confidence", decreasing = TRUE)
inspect(SortedRules_conf[1:20])

SortedRules_lift <- sort(TweetTrans_rules, by = "lift", decreasing = TRUE)
inspect(SortedRules_lift[1:20])

plot(SortedRules_sup[1:25], method = "graph", engine = "interactive")
plot(SortedRules_conf[1:25], method = "graph", engine = "interactive")
plot(SortedRules_lift[1:25], method = "graph", engine = "interactive")

plot(SortedRules_sup[1:300], jitter = 0)
plot(SortedRules_sup, method = "grouped", control = list(k = 5))
plot(SortedRules_sup[1:15], method="graph")
plot(SortedRules_sup[1:15], method="paracoord")
