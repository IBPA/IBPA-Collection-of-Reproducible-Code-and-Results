setwd("C:\\Users\\Turing\\Desktop\\Tax Project\\Paper\\Supplementary\\Figures")
library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)    # alternatively, this also loads %>%

Data = read.csv("prefilter_audit_data.csv",sep=",")
Data = Data[sapply(Data, is.numeric)]


Data <- Filter(function(x) sd(x) != 0, Data)
Data_transpose = t(Data)
Data_transpose <- Data_transpose[, !sapply(Data_transpose, function(x) { sd(x) == 0} )]

Data.Sample <- Data[sample(nrow(Data), 100), ]
#Data_transpose <- Data[, !sapply(Data, function(x) { sd(x) == 0} )]


cordata <- cor(Data)


cluster.col <- hclust(as.dist(1-cor(Data)),method = "complete")
cluster.row <- hclust(as.dist(1-cor(Data_transpose)),method = "complete")


tree.col <- cutree(cluster.col, h=0.05)
tree.row <- cutree(cluster.row, h=0.05)

dend_col <- cluster.col %>% as.dendrogram
dend_row <- cluster.row %>% as.dendrogram

library(dendextend)
plot(dend_col)
abline(h=0.05, col="red", lty=2, lwd=2)

#`%notin%` <- Negate(`%in%`)

#temp <- list()
#remove <- list()
#for(i in seq(1,length(tree.col)))
#{
#    if(tree.col[i] %notin% temp)
#      temp <- append(temp,tree.col[i])
#    else
#      remove <- cbind(remove,names(tree.col[i]))
#}

#newData <- Data[, !(names(Data) %in% remove)]
#write.csv(newData,"RemovedPCA.csv", row.names = FALSE)
#data.pca <- scale(newData, center = TRUE, scale = TRUE)
#library(devtools)
#install_github('sinhrks/ggfortify')
#library(ggfortify); library(ggplot2)
#autoplot(prcomp(newData))
library(RColorBrewer)
n <- length(unique(tree.col))
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))



library(colorspace) # get nice colors
dend_col <- as.dendrogram(hc_w2)
# Color the branches based on the tree.col:
dend_col <- color_branches(dend_col, col = col_vector, k=length(unique(tree.col))) #, groupLabels=iris_species)

# We hang the dendrogram a bit:
#dend_col <- hang.dendrogram(dend_col,hang_height=0.1)
# reduce the size of the labels:
# dend_col <- assign_values_to_leaves_nodePar(dend_col, 0.5, "lab.cex")
dend_col <- set(dend_col, "labels_cex", 0.5)
# And plot:
par(mar = c(3,3,3,7))
plot(dend_col, 
     main = "Clustered Iris data set
     (the labels give the true flower species)", 
     horiz =  TRUE,  nodePar = list(cex = .007))
legend("topleft", legend = iris_species, fill = rainbow_hcl(3))

some_col_func <- function(n) rev(colorspace::diverge_hsv(n,
                                                      h = c(1,240),
                                                      power = c(1/5, 1), 
                                                      gamma = NULL, 
                                                      fixup = TRUE, 
                                                      alpha = 1))
dend_col <- dend_col %>% set("branches_lwd", 3)
col_labels <- get_leaves_branches_col(dend_col)
col_labels <- col_labels[order(order.dendrogram(dend_col))]

library("RColorBrewer")
library("colorspace")
PCC <- gplots::heatmap.2(as.matrix(cor(Data)), 
                  main = "Pearson Correlation of Features",
                  srtCol = 50,
                  dendrogram = "both",
                  Rowv = dend_col,
                  Colv = dend_col, # this to make sure the columns are not ordered
                  trace="none",          
                  margins =c(12,22),  
                  key = TRUE,
                  keysize=1,
                  #key.par=list(mar=c(5,1,5,0)),
                  denscol = "grey",
                  density.info = "density",
                  col = some_col_func(10000)
)




