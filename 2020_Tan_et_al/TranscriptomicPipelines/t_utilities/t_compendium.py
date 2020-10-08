import sys
if (sys.version_info < (3, 0)):
    import t_metadata
    import t_data
    import t_compendium_exceptions
else:
    from . import t_metadata
    from . import t_data
    from . import t_compendium_exceptions
    
from enum import Enum
    
class TranscriptomeCompendiumConst(Enum):
    METADATA_SERIES_COLUMN = "series_id" #Please refer metadata object in t_metadata.py

class TranscriptomeCompendium:
    def __init__(self, query_string = "", query_ids = []):
        self.metadata = t_metadata.TranscriptomeMetadata(query_string = query_string, query_ids = query_ids)
        self.data = t_data.TranscriptomeData()
        
    def get_metadata(self):
        return self.metadata
        
    def set_metadata(self, metadata):
        self.metadata = metadata
        
    def get_data(self):
        return self.data
    
    
class TranscriptomeCompendiumCollections:
    def __init__(self):
        self.compendium_list = []
        self.merged_metadata_table = None
        self.merged_metadata_table_path = None
        self.merged_data_matrix = None
        self.merged_data_matrix_path = None
        self.imputation_mode = None
        self.imputed_data_matrix = None
        self.imputed_data_matrix_path = None
        self.normalization_mode = None
        self.normalized_data_matrix = None
        self.normalized_data_matrix_path = None
        
    def get_compendium_list(self):
        return self.compendium_list
        
    def set_merged_metadata_table(self, merged_metadata_table, merged_metadata_table_path):
        self.merged_metadata_table = merged_metadata_table
        self.merged_metadata_table_path = merged_metadata_table_path
        
    def get_merged_metadata_table(self):
        return self.merged_metadata_table
        
    def get_merged_metadata_series(self):
        print(TranscriptomeCompendiumConst.METADATA_SERIES_COLUMN.value)
        return self.merged_metadata_table[TranscriptomeCompendiumConst.METADATA_SERIES_COLUMN.value]
        
    def output_merged_metadata_table(self):
        try:
            self.merged_metadata_table.to_csv(self.merged_metadata_table_path)
        except Exception as e:
            raise t_compendium_exceptions.FailedToWriteMergedMetadataTable('Failed to write merged metadata table')
        
    def get_merged_data_matrix(self):
        return self.merged_data_matrix
        
    def set_merged_data_matrix(self, merged_data_matrix, merged_data_matrix_path):
        self.merged_data_matrix = merged_data_matrix
        self.merged_data_matrix_path = merged_data_matrix_path
        
    def output_merged_data_matrix(self):
        try:
            self.merged_data_matrix.to_csv(self.merged_data_matrix_path)
        except Exception as e:
            raise t_compendium_exceptions.FailedToWriteMergedDataMatrix('Failed to write merged data matrix')
        
    def get_imputed_data_matrix(self):
        return self.imputed_data_matrix
        
    def set_imputed_data_matrix(self, imputed_data_matrix, imputation_mode, imputed_data_matrix_path):
        self.imputed_data_matrix = imputed_data_matrix
        self.imputation_mode = imputation_mode
        self.imputed_data_matrix_path = imputed_data_matrix_path
        
    def output_imputed_data_matrix(self):
        try:
            self.imputed_data_matrix.to_csv(self.imputed_data_matrix_path)
        except Exception as e:
            raise t_compendium_exceptions.FailedToWriteImputedDataMatrix('Failed to write imputed data matrix')
    
    def get_normalized_data_matrix(self):
        return self.normalized_data_matrix
        
    def set_normalized_data_matrix(self, normalized_data_matrix, normalization_mode, normalized_data_matrix_path):
        self.normalized_data_matrix = normalized_data_matrix
        self.normalization_mode = normalization_mode
        self.normalized_data_matrix_path = normalized_data_matrix_path
        
    def output_normalized_data_matrix(self):
        try:
            self.normalized_data_matrix.to_csv(self.normalized_data_matrix_path)
        except Exception as e:
            raise t_compendium_exceptions.FailedToWriteNormalizedDataMatrix('Failed to write normalized data matrix')
        
    def reset_compendium_list(self):
        self.compendium_list = []
        
    def reset_merged_data(self):
        self.merged_metadata_table = None
        self.merged_data_matrix = None
        self.imputation_mode = None
        self.imputed_data_matrix = None
        self.normalization_mode = None
        self.normalized_data_matrix = None
        
    def append_additional_compendium(self, compendium_add):
        self.compendium_list.append(compendium_add)
        self.reset_merged_data()