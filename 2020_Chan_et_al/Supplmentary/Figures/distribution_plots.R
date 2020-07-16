setwd("C:\\Users\\T\\Desktop")

library(ggplot2)
library(tibble)
library(grid)
options(scipen=5)
ReturnData = read.csv("ReturnData1.txt",sep="\t")
for(i in names(ReturnData)){
  type <- ReturnData[[i]]
#  type <- ReturnData$fcurGrossSales
  d <- density(type) # returns the density data
  
  sd_data = sd(type)
  mean_data = mean(type)
  x_range = c(mean_data - sd_data*2,mean_data + sd_data*2)
  
  random.sample <- sample(length(type), 1000)
  point.plots <- tibble(x = random.sample, y = 0.0000005)
  
  outputfile <- paste("C:/Users/T/Desktop/Distribution Plots/", i, sep="")
  outputfile <- paste(outputfile, ".tiff", sep="")
  tiff(outputfile, units="cm", width=40, height=20, res=300)
  print(outputfile)

  plot.title <- paste("Distribution of ", i, sep="")
  
  xmin.cutoff <- mean_data - sd_data*2
  xmax.cutoff <- mean_data + sd_data*2
  
  x = density(ReturnData$fcurGrossSales,n=10000,from = 0, to = xmax.cutoff)$x
  y = density(ReturnData$fcurGrossSales,n=10000,from = 0, to = xmax.cutoff)$y
  
  df = data.frame(x=x,y=y)
  df2 = data.frame(x=sample(length(type), 1000), y = rep(0.000001,1000))
  
  a_plot <- ggplot(df,aes(x=x,y=y)) + 
    geom_line(size=1.2, colour = "light green") +
    xlab("\n Gross Sales ($)") + ylab("Density \n") + 
    ggtitle(plot.title) + 
    geom_vline(xintercept = mean_data, color = "red", size=0.5) 
  
  b_plot <- ggplot(df,aes(x=x,y=y)) + 
    geom_line(size=1.2, colour = "light green") +
    xlab("\n Gross Sales ($)") + ylab("Density \n") + 
    geom_vline(xintercept = xmin.cutoff, size=0.5, linetype="dotted", show.legend = TRUE) +
    geom_vline(xintercept = xmax.cutoff, size=0.5, linetype="dotted", show.legend = TRUE) +
    geom_vline(xintercept = mean_data, color = "red", size=0.5, show.legend = TRUE) +
    scale_color_manual(name = "statistics", values = c(median = "blue", mean = "red"))
  
  vp <- viewport(width = 0.39, height = 0.39, x = 0.77, y = 0.73)
  
  insert_minor <- function(major_labs, n_minor) {labs <- 
    c( sapply( major_labs, function(x) c(x, rep("", 4) ) ) )
  labs[1:(length(labs)-n_minor)]}
  
  print(a_plot +
        theme_bw() +
        theme(axis.line = element_line(colour = "black"),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              panel.border = element_blank(),
              panel.background = element_rect(colour = "black", size=1.2),
              axis.line.x = element_line(color="black", size = 0.5),
              axis.line.y = element_line(color="black", size = 0.5),
              axis.text.x=element_text(angle=45, hjust=1, size=13),
              axis.text.y=element_text(size=15),
              axis.text=element_text(size=12),
              axis.title=element_text(size=16)) +
          scale_x_continuous(expand = c(0, 0), limits = c(max(min(x),mean_data - sd_data*2), min(max(x),mean_data + sd_data*2))) +
          scale_y_continuous(expand = c(0, 0), limit = c(min(y), max(y))))
  
  print(b_plot +
        theme_bw() +
        scale_x_continuous(expand = c(0, 0), limits = c(max(min(x),mean_data - sd_data*2), max(x))) + 
        scale_y_continuous(expand = c(0, 0), limit = c(min(y), max(y))) +
        theme(axis.text.x=element_text(angle=45, hjust=1)), 
        vp = vp)
#  break
}