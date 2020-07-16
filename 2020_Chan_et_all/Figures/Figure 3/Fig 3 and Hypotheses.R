# Validate all the hypotheses
setwd("C:/Users/Turing/Desktop/Tax Project/Paper/Figures/Figure 3/Hypothesis Testing - Ameen [USED]/Code_Ameen/")
source("./h1.R")
source("./h2.R")
source("./h3.R")
source("./h4.R")
source("./h5.R")

jpeg(filename = "Figure 3.jpeg", width = 4000, height = 3466.667,
     pointsize = 0.5, quality = 10000, bg = "white", res = 250)
 
ggarrange(grossSales, INDCORP, Ratio, SalesVar, delay,
          labels = c("A", "B", "C", "D", "E"), ncol = 3, nrow = 2)

dev.off()