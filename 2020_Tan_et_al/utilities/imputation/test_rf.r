setwd("C:/Users/Bigghost/Documents/GitHub/AutomatedOmicsCompendiumPreparationPipeline/utilities/imputation")
data = read.csv("data0.0_50.csv",header=F)

library(snow)
library(missForest)
library(doParallel)
cl <- makeCluster(8,type="SOCK")
registerDoParallel(cl)
rf_result = missForest(data,verbose = T,parallelize ='variables')

