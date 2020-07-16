# h_common.R
get_aggregated_per_period <- function(df) {
  
  # Aggregate
  df_aggr <- aggregate(list(df$fcurGrossSales, df$fcurTaxableTransactions, df$fcurPenalty), 
                      by = list(df$DasID, df$AUDIT_START_DATE, df$AUDIT_END_DATE, df$AUDIT_PERIOD,
                                df$TOPLINE_TAX, df$fstrEntityType, df$fstrNAICS, df$DasCityID), 
                      sum)
  
  # Name columns
  colnames(df_aggr) <- c("DasID", "AUDIT_START_DATE", "AUDIT_END_DATE", "AUDIT_PERIOD", 
                         "TOPLINE_TAX", "fstrEntityType", "fstrNAICS", "DasCityID",
                         "fcurGrossSales", "fcurTaxableTransactions", "fcurPenalty")
  
  # Normalize
  df_aggr[,c("fcurGrossSales", "fcurTaxableTransactions", "fcurPenalty")] = df_aggr[,c("fcurGrossSales", "fcurTaxableTransactions", "fcurPenalty")]/df_aggr$AUDIT_PERIOD

  return(df_aggr)
}

get_bins <- function(df){
  return(sprintf("%s_%s",df$fstrNAICS, df$DasCityID))
}

get_thresholds <- function(){
  return(c(0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75))
}

