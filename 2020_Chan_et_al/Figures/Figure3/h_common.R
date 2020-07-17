# h_common.R
get_aggregated_per_period <- function(df) {
  
  # Aggregate
  df_aggr <- aggregate(list(df$<gross sales>, df$<taxable transactions>, df$<late penalty>), 
                      by = list(df$<ID>, df$<audit start date>, df$<audit end date>, df$<audit period>,
                                df$<audit yield>, df$<business type>, df$<NAICS Code>, df$<City ID>), 
                      sum)
  
  # Name columns
  colnames(df_aggr) <- c("ID", "audit start date", "audit end date", "audit period", 
                         "audit yield", "business type", "NAICS codes", "City ID",
                         "gross sales", "taxable transactions", "late penalty")
  
  # Normalize
  df_aggr[,c("gross sales", "taxable transactions", "late penalty")] = df_aggr[,c("gross sales", "taxable transactions", "late penalty")]/df_aggr$<audit period>

  return(df_aggr)
}

get_bins <- function(df){
  return(sprintf("%s_%s",df$<NAICS code>, df$<City ID>))
}

get_thresholds <- function(){
  return(c(0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75))
}

