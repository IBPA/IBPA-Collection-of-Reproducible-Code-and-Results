# Hypothesis 4
source("./h_common.R")

# 1) Read the data
src_filepath = file.path("./Audit_info.csv")
df <- read.table(src_filepath, sep = ",", header = TRUE)


##################################
# 3) Test hypothesis.4: 
print("*************** Hypothesis 4 ***************")

# 3.1 ) Aggregate per period with variance of <gross sales>
df <- aggregate(df[,c("<gross sales>")], 
                              list(df$<ID>, df$<audit start date>, df$<audit end date>, 
                                  df$<audit yield>, df$<NAICS Code>, df$<City ID>), var)
colnames(df) <- c("<ID>", "<audit start date>", "<audit end date>", "<audit yield>",
                                "<NAICS Code>", "<City ID>", "<gross sales>")

# 3.2) Get bins
df$bin_name <- get_bins(df)

# 3.3) Find the bin_mean per bin
df_bin_means <- aggregate(df[,c("<gross sales>")], list(df$bin_name), mean)
colnames(df_bin_means) <- c("bin_name", "bin_mean_var_gross_sales")

# 3.4) Assign the bin_means relevant for each record
df$bin_mean_var_gross_sales <- mapply(function(bin_name) {
  df_bin_means[df_bin_means$bin_name == bin_name, c("bin_mean_var_gross_sales")]
},
df[,c("bin_name")])

# 3.5) Evaluate hypothesis for each threshold
p_tresholds <- get_thresholds()
for(p in p_tresholds) {
  df_G1 <- df[df$<gross sales> < df$bin_mean_var_gross_sales * p,]
  df_G2 <- df[df$<gross sales> >= df$bin_mean_var_gross_sales * p,]
  if(p == 1.00){
    Group_Low = df_G1$<audit yield>
    Group_High = df_G2$<audit yield>
  }
  test_res <- wilcox.test(df_G1$<audit yield>, df_G2$<audit yield>, alternative = "greater", paired = FALSE)
  message(sprintf("p: %.2f, nLess: %i, nMore: %i, p-value = %0.3g, adj p-value = %0.3g",
                  p, nrow(df_G1), nrow(df_G2), test_res$p.value, p.adjust(test_res$p.value, method = "fdr", n = length(p_tresholds))))
}

Group_Low_Df = as.data.frame(Group_Low, stringsAsFactors=FALSE)
Group_Low_Df$Label = "Below Threshold"
colnames(Group_Low_Df) = c("Value", "Label")
Group_High_Df = as.data.frame(Group_High, stringsAsFactors=FALSE)
Group_High_Df$Label = "Above Threshold"
colnames(Group_High_Df) = c("Value", "Label")
Group_All_Df = as.data.frame(c(Group_High,Group_Low), stringsAsFactors=FALSE)
Group_All_Df$Label = "No Threshold/All Values"
colnames(Group_All_Df) = c("Value", "Label")

Joined=rbind(Group_Low_Df, Group_High_Df, Group_All_Df)
Joined$Label <- factor(Joined$Label , levels=c("Below Threshold", "Above Threshold", "No Threshold/All Values"))

my_comparisons <- list( c("Above Threshold","No Threshold/All Values"),
                        c("Below Threshold","Above Threshold"),
                        c("Below Threshold","No Threshold/All Values"))
stat.test = compare_means(Value ~ Label,  data = Joined, p.adjust.method = "fdr")
stat.test <- stat.test %>% mutate(y.position = c(11.7, 11.8, 11.6))

Joined = Joined[Joined$Value > 0,]

Ratio = ggplot(Joined, aes(x = Label, y = Value, fill=Label)) + 
  geom_boxjitter(jitter.shape = 21,
                 jitter.colour = NA, 
                 outlier.shape=NA,
                 outlier.colour = NULL,
                 errorbar.draw = TRUE,
                 errorbar.length=0.2) + 
  geom_signif(comparisons = list(c("Above Threshold","No Threshold/All Values")),
              annotations="n.s.",
              y_position = 190000, 
              tip_length = 0, 
              vjust=0) +
  geom_signif(comparisons = list(c("Below Threshold","Above Threshold")),
              annotations="n.s.",
              y_position = 200000, 
              tip_length = 0, 
              vjust=0.4) +
  geom_signif(comparisons = list(c("Below Threshold","No Threshold/All Values")),
              annotations="n.s.",
              y_position = 210000, 
              tip_length = 0, 
              vjust=0.4) +
  coord_cartesian(ylim = c(0, 210000)) +
  theme_classic() +
  theme(axis.line.x = element_line(color="black", size = 1),
        axis.line.y = element_line(color="black", size = 1)) +
  ggtitle(wrapper("A lower taxable sales to gross sales ratio positively influences audit yield", 
                  width=50)) +
  theme(plot.title = element_text(hjust = 0.5, size=12),
        axis.title.x=element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, size=1),
        legend.position = "none") +
  labs(y="Audit Yield ($)", 
       x = "Threshold Bins") +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10))


p_tresholds <- get_thresholds()
for(p in p_tresholds) {
  df_G1 <- df[df$<gross sales> < df$bin_mean_<gross sales> * p,]
  df_G2 <- df[df$<gross sales> >= df$bin_mean_<gross sales> * p,]
  test_res <- wilcox.test(df_G1$<audit yield>, df_G2$<audit yield>, alternative = "less", paired = FALSE)
  message(sprintf("p: %.2f, nLess: %i, nMore: %i, inv p-value = %0.3g, inv adj p-value = %0.3g",
                  p, nrow(df_G1), nrow(df_G2), test_res$p.value, p.adjust(test_res$p.value, method = "fdr", n = length(p_tresholds))))
}
