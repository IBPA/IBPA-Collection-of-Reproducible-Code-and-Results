# Hypothesis 2

source("./h_common.R")

# 1) Read the data
src_filepath = file.path("./Audit_info.csv")
df <- read.table(src_filepath, sep = ",", header = TRUE)

# 2) Aggregate for each business per period
df <- get_aggregated_per_period(df)

##################################
# Test hypothesis.2: TOPLINE_TAX is higher for for fstrEntityType =IND compared to fstrEntityType =CORP
print("*************** Hypothesis 2 ***************")
df_G1 <- df[df$fstrEntityType == "IND",]
df_G2 <- df[df$fstrEntityType == "CORP",]

Group_Low = df_G1$TOPLINE_TAX
Group_High = df_G2$TOPLINE_TAX

test_res <- wilcox.test(df_G1$TOPLINE_TAX, df_G2$TOPLINE_TAX, alternative = "greater", paired = FALSE)
test_res_inv <- wilcox.test(df_G1$TOPLINE_TAX, df_G2$TOPLINE_TAX, alternative = "less", paired = FALSE)
message(sprintf("Number of IND: %i, Number of CORP: %i, p-value = %0.3g", 
                nrow(df_G1), nrow(df_G2), test_res$p.value))
message(sprintf("Number of IND: %i, Number of CORP: %i, inv p-value = %0.3g", 
                nrow(df_G1), nrow(df_G2), test_res_inv$p.value))


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
stat.test <- stat.test %>% mutate(y.position = c(190000, 200000, 180000))

Joined = Joined[Joined$Value > 0,]

INDCORP = ggplot(Joined, aes(x = Label, y = Value, fill=Label)) + 
  geom_boxjitter(jitter.shape = 21,
                 jitter.colour = NA, 
                 outlier.shape=NA,
                 outlier.colour = NULL,
                 errorbar.draw = TRUE,
                 errorbar.length=0.2) + 
  geom_signif(comparisons = list(c("Above Threshold","No Threshold/All Values")),
              annotations="n.s.",
              y_position = 170000, 
              tip_length = 0, 
              vjust=0.4) +
  geom_signif(comparisons = list(c("Below Threshold","Above Threshold")),
              annotations="n.s.",
              y_position = 175000, 
              tip_length = 0, 
              vjust=0.4) +
  geom_signif(comparisons = list(c("Below Threshold","No Threshold/All Values")),
              annotations="n.s.",
              y_position = 185000, 
              tip_length = 0, 
              vjust=0.4) +
  coord_cartesian(ylim = c(0, 185000)) +
  theme_classic() +
  theme(axis.line.x = element_line(color="black", size = 1),
        axis.line.y = element_line(color="black", size = 1)) +
  ggtitle(wrapper("Audits on individual businesses are more likely to produce higher yield", 
                  width=40)) +
  theme(plot.title = element_text(hjust = 0.5, size=12),
        axis.title.x=element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, size=1),
        legend.position = "none") +
  labs(y="Audit Yield ($)", 
       x = "Threshold Bins") +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10))

