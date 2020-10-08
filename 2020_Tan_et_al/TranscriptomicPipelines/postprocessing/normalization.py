import sys
if (sys.version_info < (3, 0)):
    import p_module_template
else:
    from . import p_module_template

from enum import Enum
import pandas as pd

class NormalizationOptions(Enum):
    QUANTILE = "quantile"


class NormalizationParameters:
    def __init__(   self,
                    normalized_data_matrix_path = "NormalizedDataMatrix.csv"):
        self.normalization_options = NormalizationOptions.QUANTILE.value
        self.normalized_data_matrix_path = normalized_data_matrix_path
       
        
    def get_normalized_data_matrix_path(self):
        return self.normalized_data_matrix_path

    
class Normalization(p_module_template.PostprocessingSubModule):
    def __init__(self, owner):
        self.owner = owner
        self.parameters = NormalizationParameters()
        
        self.configure_parameter_set()
        
    def configure_parameter_set(self):
        parameter_set = self.get_parameter_set()
        self.parameters.normalization_options       = parameter_set.p_normalization_parameters_normalization_option
        self.parameters.normalized_data_matrix_path = parameter_set.p_normalization_parameters_normalized_data_matrix_path
        
    def normalize_data_matrix(self):
        if self.parameters.normalization_options == NormalizationOptions.QUANTILE.value:
            self.normalize_data_matrix_quantile()
            
    def normalize_data_matrix_quantile(self):
        #Reference: https://intellipaat.com/community/5641/quantile-normalization-on-pandas-dataframe
        t_compendium_collections = self.get_t_compendium_collections()
        imputed_data_matrix = t_compendium_collections.get_imputed_data_matrix()
        rank_mean = imputed_data_matrix.stack().groupby(imputed_data_matrix.rank(method='first').stack().astype(int)).mean()
        normalized_data_matrix = imputed_data_matrix.rank(method='min').stack().astype(int).map(rank_mean).unstack()
        t_compendium_collections.set_normalized_data_matrix(normalized_data_matrix, 
                                                        self.parameters.normalization_options,
                                                        self.parameters.normalized_data_matrix_path)
        t_compendium_collections.output_normalized_data_matrix()