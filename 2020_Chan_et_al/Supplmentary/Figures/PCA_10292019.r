setwd("C:/Users/Turing/Desktop")
load('AuditDataAnalysisB2.rdata')
library(Rtsne)
filtered <- read.csv(file = 'filtered_audit_data_alldata.csv')
filtered_numerical_features = which(gsub("fcur","",colnames(filtered)) != colnames(filtered))

numerical_features = which(gsub("fcur","",colnames(Return_RegistrationData)) != colnames(Return_RegistrationData))
categorical_features = which(gsub("fstr","",colnames(Return_RegistrationData)) != colnames(Return_RegistrationData))
city_features = which(gsub("City","",colnames(Return_RegistrationData)) != colnames(Return_RegistrationData))
zip_features = which(gsub("Zip","",colnames(Return_RegistrationData)) != colnames(Return_RegistrationData))
#categorical_features = c(categorical_features,city_features,zip_features)
data_categorical_part = NULL
for (i in 1 : length(categorical_features)){
    if (colnames(Return_RegistrationData)[categorical_features[i]] == "fstrSiteId")next
    if (colnames(Return_RegistrationData)[categorical_features[i]] == "fstrFormType") next
    types = unique(Return_RegistrationData[,categorical_features[i]])
    if (length(types) > 1){
        tmp_matrix = matrix(0,nrow=nrow(Return_RegistrationData),ncol=length(types))
        for (j in 1 : length(types)){
            tmp_matrix[which(Return_RegistrationData[,categorical_features[i]] == types[j]),j] = 1
        }
        colnames(tmp_matrix) = types
        data_categorical_part = cbind(data_categorical_part,tmp_matrix)
    }
}

data_numerical_part = Return_RegistrationData[,numerical_features]
filtered_data_numerical_part = filtered[,filtered_numerical_features]
filtered_data = cbind(filtered_data_numerical_part)
filtered_data_minmax = apply(filtered_data,2,function(x){(x-min(x))/(max(x)-min(x))})

data = cbind(data_numerical_part)
data_minmax = apply(data,2,function(x){(x-min(x))/(max(x)-min(x))})

data_zscore = apply(data,2,function(x){(x-mean(x))/(sd(x))})
data_normalized_grossSales = apply(data,2,function(x,grossSales){
    grossSales[grossSales < 1] = 1
    return(x/grossSales)
},data$fcurGrossSales)

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

#BiocManager::install("preprocessCore")

library(preprocessCore)
data_quantile = normalize.quantiles(as.matrix((data)))
data_quantile = (data_quantile)
dimnames(data_quantile) = dimnames(data)
data_quantile = as.data.frame(data_quantile)
data_minmax = data_quantile


library(oce)
library(tidyverse)
library(plyr)

pca_res = prcomp(data_minmax,center = F,scale. = F)
filtered_pca_res = prcomp(na.omit(filtered_data_minmax),center = F,scale. = F)
pca_filtered = as.data.frame(filtered_pca_res$x)
pca = as.data.frame(pca_res$x)

audited = filtered$CASE_ID != 0
yield = filtered$TOPLINE_TAX > 0
yield2 = filtered$TOPLINE_TAX > 20000

pca_filtered$color = c("black")
pca_filtered$color[audited] = c("blue")
pca_filtered$color[yield] = c("red")
pca_filtered$color[yield2] = c("green")

colors <- c("black"="black",
            "red"="red",
            "blue"="blue",
            "green"="green")
size = 2
main_pca <- ggplot(data=pca_filtered, aes(x = PC1, y = PC2)) + 
    geom_point(data=pca_filtered[pca_filtered$color == 'black',], aes(col="black"), size=size) + 
    geom_point(data=pca_filtered[pca_filtered$color == 'blue',], aes(col="blue"), size=size) +
    geom_point(data=pca_filtered[pca_filtered$color == 'red',], aes(col="red"), size=size) +
    geom_point(data=pca_filtered[pca_filtered$color == 'green',], aes(col="green"), size=size) +
    ggtitle("Principal Component Analysis (Minmax Normalization)") +
    xlab("Principal Component 1 (78.87%)") +
    ylab("Principal Component 2 (10.43%)") +
    scale_colour_manual(name=element_blank(),
                        label=c("All returns","Audited returns","Tax yield > $20,000","Tax yield > $0"),
                        values=colors) +
    guides(color=guide_legend(override.aes=list(fill=NA))) +
    theme(panel.background = element_blank(),
          panel.border = element_rect(colour = "black", fill=NA, size=1),
          plot.title = element_text(hjust = 0.5),
          legend.key=element_blank(),
          legend.position = c(0.01, 0.01),
          legend.justification = c(0,0))
#main_pca

audited = Return_Audit$CASE_ID != 0
yield = Return_Audit$TOPLINE_TAX > 0
yield2 = Return_Audit$TOPLINE_TAX > 20000

pca$color = c("black")
pca$color[audited] = c("blue")
pca$color[yield] = c("red")
pca$color[yield2] = c("green")

size = 1
inset_pca <- ggplot(data=pca, aes(x = PC1, y = PC2)) + 
    geom_point(data=pca[pca$color == 'black',], col="black", size=size) + 
    geom_point(data=pca[pca$color == 'blue',], col="blue", size=size) +
    geom_point(data=pca[pca$color == 'red',], col="red", size=size) +
    geom_point(data=pca[pca$color == 'green',], col="green", size=size) +
    scale_x_continuous(position = 'bottom') +
    scale_y_continuous(position = 'right') +
    theme(axis.title.x=element_blank(),
          axis.title.y=element_blank(),
          panel.background = element_blank(),
          panel.border = element_rect(colour = "black", fill=NA, size=0.7))
#inset_pca
main_pca + annotation_custom(ggplotGrob(inset_pca), 
                             xmin = -1.9, xmax = -1,
                             ymin = 0.5, ymax = 1)

#ggsave("Fig3_PCA_alldata.jpeg", dpi = 2500)

tsne = Rtsne(data_minmax,dims = 2,normalize=F, pca = FALSE, verbose=TRUE,max_iter = 500,check_duplicates=F)
filtered_tsne = Rtsne(filtered_data_minmax,dims = 2,normalize=F, pca = FALSE, verbose=TRUE,max_iter = 500,check_duplicates=F)
tsnedf = as.data.frame(tsne$Y)
filtered_tsnedf = as.data.frame(filtered_tsne$Y)
colnames(tsnedf) = c("tsne1", "tsne2")
colnames(filtered_tsnedf) = c("tsne1", "tsne2")

audited = filtered$CASE_ID != 0
yield = filtered$TOPLINE_TAX > 0
yield2 = filtered$TOPLINE_TAX > 20000

filtered_tsnedf$color = c("black")
filtered_tsnedf$color[audited] = c("blue")
filtered_tsnedf$color[yield] = c("red")
filtered_tsnedf$color[yield2] = c("green")

colors <- c("black"="black",
            "red"="red",
            "blue"="blue",
            "green"="green")

size = 2
main_tsne <- ggplot(data=filtered_tsnedf, aes(x = tsne1, y = tsne2)) + 
    geom_point(data=filtered_tsnedf[filtered_tsnedf$color == 'black',], aes(col="black"), size=size) + 
    geom_point(data=filtered_tsnedf[filtered_tsnedf$color == 'blue',], aes(col="blue"), size=size) +
    geom_point(data=filtered_tsnedf[filtered_tsnedf$color == 'red',], aes(col="red"), size=size) +
    geom_point(data=filtered_tsnedf[filtered_tsnedf$color == 'green',], aes(col="green"), size=size) +
    ggtitle("t-SNE (Minmax Normalization)") +
    xlab("t-SNE dimension 1") +
    ylab("t-SNE dimension 2") +
    xlim(-35, 20) +
    ylim(-20, 20) +
    scale_colour_manual(name=element_blank(),
                        label=c("All returns","Audited returns","Tax yield > $20,000","Tax yield > $0"),
                        values=colors) +
    guides(color=guide_legend(override.aes=list(fill=NA))) +
    theme(panel.background = element_blank(),
          panel.border = element_rect(colour = "black", fill=NA, size=1),
          plot.title = element_text(hjust = 0.5),
          legend.key=element_blank(),
          legend.position = c(0.01, 0.01),
          legend.justification = c(0,0))
main_tsne

audited = Return_Audit$CASE_ID != 0
yield = Return_Audit$TOPLINE_TAX > 0
yield2 = Return_Audit$TOPLINE_TAX > 20000

tsnedf$color = c("black")
tsnedf$color[audited] = c("blue")
tsnedf$color[yield] = c("red")
tsnedf$color[yield2] = c("green")

size = 0.001
inset_tsne <- ggplot(data=tsnedf, aes(x = tsne1, y = tsne2)) + 
    geom_point(data=tsnedf[tsnedf$color == 'black',], aes(col="black"), size=size) + 
    geom_point(data=tsnedf[tsnedf$color == 'blue',], aes(col="blue"), size=size) +
    geom_point(data=tsnedf[tsnedf$color == 'red',], aes(col="red"), size=size) +
    geom_point(data=tsnedf[tsnedf$color == 'green',], aes(col="green"), size=size) +
    scale_x_continuous(position = 'bottom') +
    scale_y_continuous(position = 'right') +
    scale_colour_manual(name=element_blank(),
                        values=colors) +
    theme(axis.title.x=element_blank(),
          axis.title.y=element_blank(),
          legend.position="none",
          panel.background = element_blank(),
          panel.border = element_rect(colour = "black", fill=NA, size=0.7))
inset_tsne

main_tsne + annotation_custom(ggplotGrob(inset_tsne), 
                              xmin = -37.7, xmax = -17,
                              ymin = 3, ymax = 22)

#ggsave("Fig3_tSNE_alldata.jpeg", dpi = 2500)


#ONLY AUDIT DATA
filtered <- read.csv(file = 'filtered_audit_data.csv')
filtered_numerical_features = which(gsub("fcur","",colnames(filtered)) != colnames(filtered))
Return_RegistrationData <- read.csv(file = 'prefilter_audit_data.csv')
numerical_features = which(gsub("fcur","",colnames(Return_RegistrationData)) != colnames(Return_RegistrationData))
categorical_features = which(gsub("fstr","",colnames(Return_RegistrationData)) != colnames(Return_RegistrationData))
city_features = which(gsub("City","",colnames(Return_RegistrationData)) != colnames(Return_RegistrationData))
zip_features = which(gsub("Zip","",colnames(Return_RegistrationData)) != colnames(Return_RegistrationData))
#categorical_features = c(categorical_features,city_features,zip_features)
data_categorical_part = NULL
for (i in 1 : length(categorical_features)){
    if (colnames(Return_RegistrationData)[categorical_features[i]] == "fstrSiteId")next
    if (colnames(Return_RegistrationData)[categorical_features[i]] == "fstrFormType") next
    types = unique(Return_RegistrationData[,categorical_features[i]])
    if (length(types) > 1){
        tmp_matrix = matrix(0,nrow=nrow(Return_RegistrationData),ncol=length(types))
        for (j in 1 : length(types)){
            tmp_matrix[which(Return_RegistrationData[,categorical_features[i]] == types[j]),j] = 1
        }
        colnames(tmp_matrix) = types
        data_categorical_part = cbind(data_categorical_part,tmp_matrix)
    }
}

data_numerical_part = Return_RegistrationData[,numerical_features]
filtered_data_numerical_part = filtered[,filtered_numerical_features]
filtered_data = cbind(filtered_data_numerical_part)
data = cbind(data_numerical_part)

data_minmax = apply(data,2,function(x){(x-min(x))/(max(x)-min(x))})
filtered_data_minmax = apply(filtered_data,2,function(x){(x-min(x))/(max(x)-min(x))})



library(oce)
library(preprocessCore)
library(tidyverse)
library(plyr)

pca_res = prcomp(data_minmax,center = F,scale. = F)
filtered_pca_res = prcomp(na.omit(filtered_data_minmax),center = F,scale. = F)
pca_filtered = as.data.frame(filtered_pca_res$x)
pca = as.data.frame(pca_res$x)

audited = filtered$CASE_ID != 0
yield = filtered$TOPLINE_TAX > 0
yield2 = filtered$TOPLINE_TAX > 20000
pca_filtered$color = c("red")
pca_filtered$color[yield2] = c("green")

colors <- c("red3","turquoise4")

size = 1
main_pca <- ggplot(data=pca_filtered, aes(x = PC1, y = PC2)) + 
    geom_point(data=pca_filtered[pca_filtered$color == 'red',], aes(col="red"), size=size) +
    geom_point(data=pca_filtered[pca_filtered$color == 'green',], aes(col="green"), size=size) +
    ggtitle("Principal Component Analysis (Minmax Normalization)") +
    xlab("Principal Component 1 (65.59%)") +
    ylab("Principal Component 2 (15.65%)") +
    scale_colour_manual(name=element_blank(),
                        label=c("Tax yield \u2265 $20,000","Tax yield < $20,000"),
                        values=colors) +
    guides(color=guide_legend(override.aes=list(fill=NA))) +
    theme(panel.background = element_blank(),
          panel.border = element_rect(colour = "black", fill=NA, size=1),
          plot.title = element_text(hjust = 0.5, size=18),
          axis.title.x=element_text(size=12),
          axis.title.y=element_text(size=12),
          legend.key=element_blank(),
          legend.text = element_text(size=15),
          legend.position = c(0.99, 0.99),
          legend.justification = c(1,1))
main_pca

audited = Return_RegistrationData$CASE_ID != 0
yield = Return_RegistrationData$TOPLINE_TAX > 0
yield2 = Return_RegistrationData$TOPLINE_TAX > 20000
pca$color = c("red")
pca$color[yield2] = c("green")

colors <- c("red3","turquoise4")

size = 0.1
inset_pca <- ggplot(data=pca, aes(x = PC1, y = PC2)) + 
    geom_point(data=pca[pca$color == 'red',], aes(col="red"), size=size) +
    geom_point(data=pca[pca$color == 'green',],aes(col="green"), size=size) +
    scale_x_continuous(position = 'top') +
    scale_y_continuous(position = 'left') +
    scale_colour_manual(name=element_blank(),
                        values=colors) +
    labs(title="Before Normalization") +
    theme(plot.title=element_text(hjust=0.5, size=12),
          axis.title.x=element_blank(),
          axis.title.y=element_blank(),
          legend.position="none",
          panel.background = element_blank(),
          panel.border = element_rect(colour = "black", fill=NA, size=0.7))
#inset_pca

minmaxpca = main_pca + annotation_custom(ggplotGrob(inset_pca), 
                             xmin = -0.9, xmax = 0,
                             ymin = -1.80, ymax = -0.65)


tsne = Rtsne(data_minmax,dims = 2,normalize=F, pca = FALSE, verbose=TRUE,max_iter = 500,check_duplicates=F)
filtered_tsne = Rtsne(filtered_data_minmax,dims = 2,normalize=F, pca = FALSE, verbose=TRUE,max_iter = 500,check_duplicates=F)
tsnedf = as.data.frame(tsne$Y)
filtered_tsnedf = as.data.frame(filtered_tsne$Y)
colnames(tsnedf) = c("tsne1", "tsne2")
colnames(filtered_tsnedf) = c("tsne1", "tsne2")

audited = filtered$CASE_ID != 0
yield = filtered$TOPLINE_TAX > 0
yield2 = filtered$TOPLINE_TAX > 20000

filtered_tsnedf$color = c("red")
filtered_tsnedf$color[yield2] = c("green")

colors <- c("red3","turquoise4")

size = 1
main_tsne <- ggplot(data=filtered_tsnedf, aes(x = tsne1, y = tsne2)) + 
    geom_point(data=filtered_tsnedf[filtered_tsnedf$color == 'red',], aes(col="red"), size=size) +
    geom_point(data=filtered_tsnedf[filtered_tsnedf$color == 'green',], aes(col="green"), size=size) +
    ggtitle("t-SNE (Minmax Normalization)") +
    xlab("t-SNE dimension 1") +
    ylab("t-SNE dimension 2") +
    scale_colour_manual(name=element_blank(),
                        label=c("Tax yield \u2265 $20,000","Tax yield < $20,000"),
                        values=colors) +
    xlim(-40, 30) +
    ylim(-20, 45) +
    guides(color=guide_legend(override.aes=list(fill=NA))) +
    theme(panel.background = element_blank(),
          panel.border = element_rect(colour = "black", fill=NA, size=1),
          plot.title = element_text(hjust = 0.5, size=18),
          legend.key=element_blank(),
          axis.title.x=element_text(size=12),
          axis.title.y=element_text(size=12),
          legend.position = c(0.99, 0.99),
          legend.text = element_text(size=15),
          legend.justification = c(1,1))
main_tsne

audited = Return_RegistrationData$CASE_ID != 0
yield = Return_RegistrationData$TOPLINE_TAX > 0
yield2 = Return_RegistrationData$TOPLINE_TAX > 20000

tsnedf$color = c("red")
tsnedf$color[yield2] = c("green")

colors <- c("red3","turquoise4")

size = 0.1
inset_tsne <- ggplot(data=tsnedf, aes(x = tsne1, y = tsne2)) + 
    geom_point(data=tsnedf[tsnedf$color == 'red',], aes(col="red"), size=size) +
    geom_point(data=tsnedf[tsnedf$color == 'green',], aes(col="green"), size=size) +
    scale_x_continuous(position = 'bottom') +
    scale_y_continuous(position = 'right') +
    scale_colour_manual(name=element_blank(),
                        values=colors) +
    xlab("Before Normalization") +
    theme(axis.title.x=element_text(size=12),
          axis.title.y=element_blank(),
          legend.position="none",
          panel.background = element_blank(),
          panel.border = element_rect(colour = "black", fill=NA, size=0.7))
inset_tsne

minmaxtsne = main_tsne + annotation_custom(ggplotGrob(inset_tsne), 
                              xmin = -43, xmax = -17,
                              ymin =18, ymax = 47.5)


jpeg(filename = "Figure_2.jpeg", width = 3500, height = 1450,
     pointsize = 1, quality = 1000, bg = "white", res = 250)

ggarrange(minmaxpca, minmaxtsne, 
          heights = c(850, 850), widths =c(1000, 1000),  align = "v",
          ncol = 2, nrow = 1)
dev.off()





















data_zscore = apply(data,2,function(x){(x-mean(x))/(sd(x))})
data_normalized_grossSales = apply(data,2,function(x,grossSales){
    grossSales[grossSales < 1] = 1
    return(x/grossSales)
},data$fcurGrossSales)

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

#BiocManager::install("preprocessCore")

pca_res = prcomp(data,center = F,scale. = F)
plot(pca_res$x[,1],pca_res$x[,2],pch=20,main="PCA (without Normalization)",
     xlab=paste0("PC 1 (",format(summary(pca_res)[[6]][2,1]*100,digits = 2),"%)"),
     ylab=paste0("PC 2 (",format(summary(pca_res)[[6]][2,2]*100,digits = 2),"%)"))

audited = Return_Audit$CASE_ID != 0
yield = Return_Audit$TOPLINE_TAX > 0
yield2 = Return_Audit$TOPLINE_TAX > 20000
points(pca_res$x[audited,1],pca_res$x[audited,2],col=3,pch=20)
points(pca_res$x[yield,1],pca_res$x[yield,2],col=2,pch=20)
points(pca_res$x[yield2,1],pca_res$x[yield2,2],col=4,pch=20)
legend("topright",c("All returns","Audited returns","Tax yield > 0","Tax yield > 20K"),col=c(1,3,2,4),pch=20,bty="n",cex=0.8)


library(RColorBrewer)
palette(brewer.pal(n = 8, name = "Set1"))
tmp = palette()
palette(c("#000000",tmp))
tsne = Rtsne(data,dims = 2,normalize=F, pca = FALSE, verbose=TRUE,max_iter = 500,check_duplicates=F)
plot(tsne$Y,pch=20,main = "t-SNE (without Normalization)",col=1, xlab="tsne dim 1",ylab = "tsne dim 2",xlim=c(-25,20))
points(tsne$Y[audited,],col=3,pch=20)
points(tsne$Y[yield,],col=2,pch=20)
points(tsne$Y[yield2,],col=4,pch=20)
legend("topleft",c("All returns","Audited returns","Tax yield > 0","Tax yield > 20K"),col=c(1,3,2,4),pch=20,bty="n",cex=0.8)









pca_res = prcomp(data_zscore,center = F,scale. = F)
plot(pca_res$x[,1],pca_res$x[,2],pch=20,main="PCA (Z-Score Normalization)",
     xlab=paste0("PC 1 (",format(summary(pca_res)[[6]][2,1]*100,digits = 2),"%)"),
     ylab=paste0("PC 2 (",format(summary(pca_res)[[6]][2,2]*100,digits = 2),"%)"))

audited = Return_Audit$CASE_ID != 0
yield2 = Return_Audit$TOPLINE_TAX > 20000
points(pca_res$x[audited,1],pca_res$x[audited,2],col=3,pch=20)
points(pca_res$x[yield,1],pca_res$x[yield,2],col=2,pch=20)
points(pca_res$x[yield2,1],pca_res$x[yield2,2],col=4,pch=20)
legend("topright",c("All returns","Audited returns","Tax yield > 0","Tax yield > 20K"),col=c(1,3,2,4),pch=20,bty="n",cex=0.8)


library(RColorBrewer)
palette(brewer.pal(n = 8, name = "Set1"))
tmp = palette()
palette(c("#000000",tmp))
tsne = Rtsne(data_zscore,dims = 2,normalize=F, pca = FALSE, verbose=TRUE,max_iter = 500,check_duplicates=F)
plot(tsne$Y,pch=20,main = "t-SNE (Z-Score Normalization)",col=1, xlab="tsne dim 1",ylab = "tsne dim 2",xlim=c(-25,20))
points(tsne$Y[audited,],col=3,pch=20)
points(tsne$Y[yield,],col=2,pch=20)
points(tsne$Y[yield2,],col=4,pch=20)
legend("topleft",c("All returns","Audited returns","Tax yield > 0","Tax yield > 20K"),col=c(1,3,2,4),pch=20,bty="n",cex=0.8)


pca_res = prcomp(data_quantile,center = F,scale. = F)
plot(pca_res$x[,1],pca_res$x[,2],pch=20,main="PCA (Quantile Normalization)",
     xlab=paste0("PC 1 (",format(summary(pca_res)[[6]][2,1]*100,digits = 2),"%)"),
     ylab=paste0("PC 2 (",format(summary(pca_res)[[6]][2,2]*100,digits = 2),"%)"))

audited = Return_Audit$CASE_ID != 0
yield = Return_Audit$TOPLINE_TAX > 0
yield2 = Return_Audit$TOPLINE_TAX > 20000
points(pca_res$x[audited,1],pca_res$x[audited,2],col=3,pch=20)
points(pca_res$x[yield,1],pca_res$x[yield,2],col=2,pch=20)
points(pca_res$x[yield2,1],pca_res$x[yield2,2],col=4,pch=20)
legend("topright",c("All returns","Audited returns","Tax yield > 0","Tax yield > 20K"),col=c(1,3,2,4),pch=20,bty="n",cex=0.8)


library(RColorBrewer)
palette(brewer.pal(n = 8, name = "Set1"))
tmp = palette()
palette(c("#000000",tmp))
tsne = Rtsne(data_quantile,dims = 2,normalize=F, pca = FALSE, verbose=TRUE,max_iter = 500,check_duplicates=F)
plot(tsne$Y,pch=20,main = "t-SNE (Quantile Normalization)",col=1, xlab="tsne dim 1",ylab = "tsne dim 2",xlim=c(-25,20))
points(tsne$Y[audited,],col=3,pch=20)
points(tsne$Y[yield,],col=2,pch=20)
yield2 = Return_Audit$TOPLINE_TAX > 20000
points(tsne$Y[yield2,],col=4,pch=20)
legend("topleft",c("All returns","Audited returns","Tax yield > 0","Tax yield > 20K"),col=c(1,3,2,4),pch=20,bty="n",cex=0.8)



library(clusternor)
xmeans_res = Xmeans(tsne$Y,200)

xmeans_res$cluster
plot(tsne$Y,pch=20,col=xmeans_res$cluster)
plot(tsne$Y,pch=20,col=1)

pval_cluster = numeric(max(xmeans_res$cluster))
hit_cluster = numeric(max(xmeans_res$cluster))
found_cluster = numeric(max(xmeans_res$cluster))
for (i in 1 : max(xmeans_res$cluster)){
    tmp = xmeans_res$cluster == i
    
    tmp_audited = audited & tmp
    tmp_yield = yield & tmp
    #white: hit
    #black: normal
    q = length(which(tmp_yield==T)) #white balls drawn from the urn without replacement 
    m = length(which(yield==T)) #white balls in the urn
    n = length(which(yield==F & audited==T)) #black balls in the urn
    k = length(which(tmp_audited==T)) #balls drawn from the urn
    
    pval_cluster[i] = phyper(q,m,n,k,lower.tail = F)
    hit_cluster[i] = q
    found_cluster[i] = k
}

res = rbind(pval_cluster,hit_cluster,found_cluster,hit_cluster/found_cluster)
colnames(res) = 1:ncol(res)
res = res[,order(res[1,])]

tmp = xmeans_res$cluster == 18

tmp_audited = audited & tmp
tmp_yield = yield & tmp
tmp_yield2 = yield2 & tmp

plot(tsne$Y,pch=20,main = "tsne (Quantile Normalization)",col=1, xlab="tsne dim 1",ylab = "tsne dim 2")
points(tsne$Y[tmp,],col=5,pch=20)
points(tsne$Y[tmp_audited,],col=3,pch=20)
points(tsne$Y[tmp_yield,],col=2,pch=20)
points(tsne$Y[tmp_yield2,],col=4,pch=20)

plot(tsne$Y[tmp,],col=5,pch=20)
points(tsne$Y[tmp_audited,],col=3,pch=20)
points(tsne$Y[tmp_yield,],col=2,pch=20)
points(tsne$Y[tmp_yield2,],col=4,pch=20)

write.csv(cbind(Return_Audit,xmeans_res$cluster),"Return_Audit_Cluster.csv")

pval_col = numeric(length(numerical_features))
mean_out = numeric(length(numerical_features))
mean_in = numeric(length(numerical_features))
mean_out_raw = numeric(length(numerical_features))
mean_in_raw = numeric(length(numerical_features))
for(i in 1 : length(numerical_features)){
    
    val = data_quantile[,i]
    val_raw = data_normalized_grossSales[,i]
    boxplot(val_raw[!tmp],val_raw[tmp],main=colnames(data_quantile)[i],ylim=c(0,1))
    pval_col[i] = wilcox.test(val_raw[!tmp],val_raw[tmp],exact = T)[[3]]
    mean_out[i] = mean(val[!tmp])
    mean_in[i] = mean(val[tmp])
    mean_out_raw[i] = mean(val_raw[!tmp])
    mean_in_raw[i] = mean(val_raw[tmp])
}
res_comp = rbind(pval_col,mean_out,mean_in,mean_out_raw,mean_in_raw)
colnames(res_comp) = colnames(data_quantile)