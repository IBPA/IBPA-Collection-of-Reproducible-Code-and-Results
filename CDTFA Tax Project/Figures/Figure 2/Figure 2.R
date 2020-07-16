setwd("C:/Users/Turing/Desktop")
load('AuditDataAnalysisB2.rdata')
library(Rtsne)
library(oce)
library(tidyverse)
library(plyr)

Return_Audit = Return_Audit[<audited data>]
Return_Audit = Return_Audit[<positive audit data>]
filtered <- read.csv(file = 'data.csv')

numerical_features = <numerical features>
categorical_features = <categorical features>
city_features = <city features>
zip_features = <zip code features>
#categorical_features = c(categorical_features,city_features,zip_features)
data_categorical_part = NULL

for (i in 1 : length(categorical_features)){
  if (colnames(Return_Audit)[categorical_features[i]] == "Locaiton ID")next
  if (colnames(Return_Audit)[categorical_features[i]] == "Form Type") next
  types = unique(Return_Audit[,categorical_features[i]])
  if (length(types) > 1){
    tmp_matrix = matrix(0,nrow=nrow(Return_Audit),ncol=length(types))
    for (j in 1 : length(types)){
      tmp_matrix[which(Return_Audit[,categorical_features[i]] == types[j]),j] = 1
    }
    colnames(tmp_matrix) = types
    data_categorical_part = cbind(data_categorical_part,tmp_matrix)
  }
}

data_numerical_part = Return_Audit[,numerical_features]
ID = Return_Audit$ID
yield = Return_Audit$audit_yield
data = cbind(ID, data_numerical_part)
data_yield = cbind(ID, data_numerical_part, yield)

myfunc <- function(vec){
  sum(vec)/length(vec)
}
data = aggregate(.~ID, data, FUN=function(vec) myfunc(vec))
data_yield = aggregate(.~ID, data_yield, FUN=function(vec) myfunc(vec))
  
drop <- c(<list of features to be dropped>)
data = data[ ,!(names(data) %in% drop)]
data_yield = data_yield[ ,!(names(data_yield) %in% drop)]

remove = c(42, 137, 271, 295)
data = data[-remove,]
data_yield = data_yield[-remove,]

#write.csv(data, "pca.csv")

drop_zero <- c(<features with too many 0s>)

data = data[ ,!(names(data) %in% drop_zero)]
data_yield = data_yield[ ,!(names(data_yield) %in% drop_zero)]

data_norm = apply(data,2,function(x){(x-min(x))/(max(x)-min(x))})
#data_norm = apply(data,2,function(x){(x-mean(x))/(sd(x))})
library(preprocessCore)
#data_norm = normalize.quantiles(as.matrix((data)))

pca_res = prcomp(data_norm,center = F,scale. = F)
pca = as.data.frame(pca_res$x)
tsne = Rtsne(data_norm,dims = 2,normalize=F, pca = FALSE, verbose=TRUE,max_iter = 500,check_duplicates=F)
tsnedf = as.data.frame(tsne$Y)
colnames(tsnedf) = c("tsne1", "tsne2")

library(factoextra)
library(cluster)
library(ggpubr)

use = pca_res$x

d <- dist(use, method = "euclidean")
# Ward's method
hc5 <- hclust(d, method = "ward.D2" )

elbow = fviz_nbclust(use, FUN = hcut, k.max = 15, method = "wss")
#fviz_nbclust(use, FUN = hcut, k.max = 15, method = "silhouette")
gap_stat <- clusGap(use, FUN = hcut, nstart = 25, K.max = 15, B = 30)
gs = fviz_gap_stat(gap_stat)


num_clusters = 7

sub_grp <- cutree(hc5, k = num_clusters)
pca$color = sub_grp
tsnedf$color = sub_grp

table(sub_grp)

size = 2
main_pca <- ggplot(data=pca, aes(x = PC1, y = PC2)) + 
  geom_point(data=pca, aes(col=as.factor(color)), size=size) + 
  ggtitle("Principal Component Analysis (Minmax Normalization)") +
  xlab("Principal Component 1 (94.10%)") +
  ylab("Principal Component 2 (4.65%)") +
  scale_colour_manual(name=element_blank(),
                      label=c("Cluster 1",
                              "Cluster 2",
                              "Cluster 3",                              
                              "Cluster 4",
                              "Cluster 5",
                              "Cluster 6",
                              "Cluster 7"),
                      values=brewer.pal(n = num_clusters, name = 'Paired')) +
  guides(color=guide_legend(override.aes=list(fill=NA))) +
  theme(panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, size=1),
        plot.title = element_text(hjust = 0.5),
        legend.key=element_blank(),
        legend.position = c(0.01, 0.01),
        legend.justification = c(0,0))
main_pca

main_tsne <- ggplot(data=tsnedf, aes(x = tsne1, y = tsne2)) + 
  geom_point(data=tsnedf, aes(col=as.factor(color)), size=size) + 
  ggtitle("t-SNE (Minmax Normalization)") +
  xlab("t-SNE dimension 1") +
  ylab("t-SNE dimension 2") +
  scale_x_continuous(position = 'bottom') +
  scale_y_continuous(position = 'right') +
  scale_colour_manual(name=element_blank(),
                      label=c("Cluster 1",
                              "Cluster 2",
                              "Cluster 3",                              
                              "Cluster 4",
                              "Cluster 5",
                              "Cluster 6",
                              "Cluster 7"),
                      values=brewer.pal(n = num_clusters, name = 'Paired')) +
  guides(color=guide_legend(override.aes=list(fill=NA))) +
  theme(panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, size=1),
        plot.title = element_text(hjust = 0.5),
        legend.key=element_blank(),
        legend.position = c(0.01, 0.01),
        legend.justification = c(0,0))

main_tsne


jpeg(filename = "Figure 2.jpeg", width = 5000, height = 4000,
     pointsize = 1, quality = 1000, bg = "white", res = 300)

ggarrange(elbow, gs, main_pca, main_tsne,
          ncol = 2, nrow = 2)
dev.off()


data_df = as.data.frame(data)
data_df$cluster = sub_grp

data_yield_df = as.data.frame(data_yield)
data_yield_df$cluster = sub_grp

cluster1_yield = data_yield_df[data_yield_df$cluster == 1,]
cluster2_yield = data_yield_df[data_yield_df$cluster == 2,]
cluster3_yield = data_yield_df[data_yield_df$cluster == 3,]
cluster4_yield = data_yield_df[data_yield_df$cluster == 4,]
cluster5_yield = data_yield_df[data_yield_df$cluster == 5,]
cluster6_yield = data_yield_df[data_yield_df$cluster == 6,]
cluster7_yield = data_yield_df[data_yield_df$cluster == 7,]

y1 = dim(cluster1_yield[cluster1_yield$yield >= 20000,])[1]/dim(data_yield[data_yield_df$cluster == 1,])[1]
y2 = dim(cluster2_yield[cluster2_yield$yield >= 20000,])[1]/dim(data_yield[data_yield_df$cluster == 2,])[1]
y3 = dim(cluster3_yield[cluster3_yield$yield >= 20000,])[1]/dim(data_yield[data_yield_df$cluster == 3,])[1]
y4 = dim(cluster4_yield[cluster4_yield$yield >= 20000,])[1]/dim(data_yield[data_yield_df$cluster == 4,])[1]
y5 = dim(cluster5_yield[cluster5_yield$yield >= 20000,])[1]/dim(data_yield[data_yield_df$cluster == 5,])[1]
y6 = dim(cluster6_yield[cluster6_yield$yield >= 20000,])[1]/dim(data_yield[data_yield_df$cluster == 6,])[1]
y7 = dim(cluster7_yield[cluster7_yield$yield >= 20000,])[1]/dim(data_yield[data_yield_df$cluster == 7,])[1]

mean(cbind(y1,y2,y3,y4,y5,y6,y7))


library(arules)
library("arules")

cluster_1 = data_df[data_df$cluster == 1,][ ,!(names(data_df[data_df$cluster == 1,]) %in% c('cluster'))]
cluster_2 = data_df[data_df$cluster == 2,][ ,!(names(data_df[data_df$cluster == 2,]) %in% c('cluster'))]
cluster_3 = data_df[data_df$cluster == 3,][ ,!(names(data_df[data_df$cluster == 3,]) %in% c('cluster'))]
cluster_4 = data_df[data_df$cluster == 4,][ ,!(names(data_df[data_df$cluster == 4,]) %in% c('cluster'))]
cluster_5 = data_df[data_df$cluster == 5,][ ,!(names(data_df[data_df$cluster == 5,]) %in% c('cluster'))]
cluster_6 = data_df[data_df$cluster == 6,][ ,!(names(data_df[data_df$cluster == 6,]) %in% c('cluster'))]
cluster_7 = data_df[data_df$cluster == 7,][ ,!(names(data_df[data_df$cluster == 6,]) %in% c('cluster'))]

cluster_1 = as.data.frame(cluster_1)
cluster_2 = as.data.frame(cluster_2)
cluster_3 = as.data.frame(cluster_3)
cluster_4 = as.data.frame(cluster_4)
cluster_5 = as.data.frame(cluster_5)
cluster_6 = as.data.frame(cluster_6)
cluster_7 = as.data.frame(cluster_7)

for(i in 1:8) cluster_1[,i] <- discretize(cluster_1[,i],  "frequency", breaks=100)
for(i in 1:8) cluster_2[,i] <- discretize(cluster_2[,i],  "frequency", breaks=100)
for(i in 1:8) cluster_3[,i] <- discretize(cluster_3[,i],  "frequency", breaks=100)
for(i in 1:8) cluster_4[,i] <- discretize(cluster_4[,i],  "frequency", breaks=100)
for(i in 1:8) cluster_5[,i] <- discretize(cluster_5[,i],  "frequency", breaks=100)
for(i in 1:8) cluster_6[,i] <- discretize(cluster_6[,i],  "frequency", breaks=100)
for(i in 1:8) cluster_7[,i] <- discretize(cluster_7[,i],  "frequency", breaks=100)

cluster1_rules <- apriori(cluster_1, parameter = list(support = 0.01, confidence = 0.90))
cluster2_rules <- apriori(cluster_2, parameter = list(support = 0.01, confidence = 0.90))
cluster3_rules <- apriori(cluster_3, parameter = list(support = 0.01, confidence = 0.90))
cluster4_rules <- apriori(cluster_4, parameter = list(support = 0.01, confidence = 0.90))
cluster5_rules <- apriori(cluster_5, parameter = list(support = 0.01, confidence = 0.90))
cluster6_rules <- apriori(cluster_6, parameter = list(support = 0.01, confidence = 0.90))
cluster7_rules <- apriori(cluster_7, parameter = list(support = 0.01, confidence = 0.90))


remove_redundant <- function(rules){
  gi <- generatingItemsets(rules)
  d <- which(duplicated(gi))
  return(rules[-d])
}

cluster1_rules <- remove_redundant(cluster1_rules)
cluster2_rules <- remove_redundant(cluster2_rules)
cluster3_rules <- remove_redundant(cluster3_rules)
cluster4_rules <- remove_redundant(cluster4_rules)
cluster5_rules <- remove_redundant(cluster5_rules)
cluster6_rules <- remove_redundant(cluster6_rules)
cluster7_rules <- remove_redundant(cluster7_rules)

cluster1_rules
cluster2_rules
cluster3_rules
cluster4_rules
cluster5_rules
cluster6_rules
cluster7_rules

inspect(head(sort(cluster1_rules, by = "confidence"), 5000))
inspect(head(sort(cluster2_rules, by = "confidence"), 5000))
inspect(head(sort(cluster3_rules, by = "confidence"), 5000))
inspect(head(sort(cluster4_rules, by = "confidence"), 5000))
inspect(head(sort(cluster5_rules, by = "confidence"), 5000))
inspect(head(sort(cluster6_rules, by = "confidence"), 5000))
inspect(head(sort(cluster7_rules, by = "confidence"), 5000))

write(cluster1_rules, file = "cluster1_rules.csv", sep = ",", quote = TRUE, row.names = FALSE)
write(cluster2_rules, file = "cluster2_rules.csv", sep = ",", quote = TRUE, row.names = FALSE)
write(cluster3_rules, file = "cluster3_rules.csv", sep = ",", quote = TRUE, row.names = FALSE)
write(cluster4_rules, file = "cluster4_rules.csv", sep = ",", quote = TRUE, row.names = FALSE)
write(cluster5_rules, file = "cluster5_rules.csv", sep = ",", quote = TRUE, row.names = FALSE)
write(cluster6_rules, file = "cluster6_rules.csv", sep = ",", quote = TRUE, row.names = FALSE)
write(cluster7_rules, file = "cluster7_rules.csv", sep = ",", quote = TRUE, row.names = FALSE)



# Determine number of clusters
wss <- (nrow(data_minmax)-1)*sum(apply(data_minmax,2,var))
for (i in 2:30) wss[i] <- sum(kmeans(data_minmax,
                                     iter.max=1000,
                                     nstart=25,
                                     centers=i)$withinss)
plot(1:30, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")
abline(v = num_clusters, col="red", lwd=3, lty=2)
k <- kmeans(pca_res$x, num_clusters, nstart=25, iter.max=1000)

library(RColorBrewer)
pca$color = k$cluster
tsnedf$color = k$cluster

