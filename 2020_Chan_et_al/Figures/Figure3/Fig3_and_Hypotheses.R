# Validate all the hypotheses
source("./hyp1.R")
source("./hyp2.R")
source("./hyp3.R")
source("./hyp4.R")
source("./hyp5.R")

jpeg(filename = "Figure3.jpeg", width = 4000, height = 3466.667,
     pointsize = 0.5, quality = 10000, bg = "white", res = 250)
 
ggarrange(grossSales, INDCORP, Ratio, SalesVar, delay,
          labels = c("A", "B", "C", "D", "E"), ncol = 3, nrow = 2)

dev.off()