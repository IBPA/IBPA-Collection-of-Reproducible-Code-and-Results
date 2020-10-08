import sys
if (sys.version_info < (3, 0)):
    import p_module_template
else:
    from . import p_module_template
    
#RF Impute
#NOTE: Later this part should become a independent package so that we should import it directly
sys.path.insert(0,"../utilities/imputation")
sys.path.insert(0,"../../utilities/imputation")
import rfimpute

from enum import Enum
import pandas as pd
import numpy as np

import os

class ImputationOptions(Enum):
    AVERAGE = "average"
    ZERO    = "zero"
    KNN     = "knn"
    RF      = "random_forest"


class ImputationParameters:
    def __init__(   self,
                    imputed_data_matrix_path = "ImputedDataMatrix.csv"):
        self.imputation_options = ImputationOptions.RF.value
        self.imputed_data_matrix_path = imputed_data_matrix_path
        self.skip_imputation = True
       
    def get_imputed_data_matrix_path(self):
        return self.imputed_data_matrix_path

    
class Imputation(p_module_template.PostprocessingSubModule):
    def __init__(self, owner):
        self.owner = owner
        self.parameters = ImputationParameters()
        self.rfimpute = rfimpute.MissForestImputation()
        
        self.configure_parameter_set()
        self.configure_rfimpute_parameter_set()
        
    def configure_parameter_set(self):
        parameter_set = self.get_parameter_set()
        self.parameters.imputation_options          = parameter_set.p_imputation_parameters_imputation_option
        self.parameters.imputed_data_matrix_path    = parameter_set.p_imputation_parameters_imputed_data_matrix_path
        self.parameters.skip_imputation             = parameter_set.p_imputation_parameters_skip_imputation
        
    def configure_rfimpute_parameter_set(self):
        parameter_set = self.get_parameter_set()
        self.rfimpute.parameters.initial_guess_mode                 = parameter_set.p_imputation_rfimpute_parameters_initial_guess_option
        self.rfimpute.parameters.parallel_options                   = parameter_set.p_imputation_rfimpute_parallel_parameters_parallel_mode
        self.rfimpute.parameters.max_iter                           = parameter_set.p_imputation_rfimpute_parameters_max_iter
        self.rfimpute.parameters.num_node                           = parameter_set.p_imputation_rfimpute_parallel_parameters_n_jobs
        self.rfimpute.parameters.num_feature_local                  = parameter_set.p_imputation_rfimpute_parallel_parameters_n_feature_local
        self.rfimpute.parameters.num_core_local                     = parameter_set.p_imputation_rfimpute_parallel_parameters_n_core_local
        self.rfimpute.parameters.tmp_X_file                         = parameter_set.p_imputation_rfimpute_parallel_parameters_slurm_tmp_X_file
        
        self.rfimpute.parameters.slurm_parameters.par_num_node                  = parameter_set.constants.parallel_slurm_parameters_par_num_node
        self.rfimpute.parameters.slurm_parameters.num_node                      = 1 #Should always be 1
        self.rfimpute.parameters.slurm_parameters.par_num_core_each_node        = parameter_set.constants.parallel_slurm_parameters_par_num_core_each_node
        self.rfimpute.parameters.slurm_parameters.num_core_each_node            = parameter_set.p_imputation_rfimpute_parallel_parameters_n_core_local
        self.rfimpute.parameters.slurm_parameters.par_time_limit                = parameter_set.constants.parallel_slurm_parameters_par_time_limit
        self.rfimpute.parameters.slurm_parameters.time_limit_hr                 = parameter_set.p_imputation_rfimpute_parallel_parameters_slurm_time_limit_hr
        self.rfimpute.parameters.slurm_parameters.time_limit_min                = parameter_set.p_imputation_rfimpute_parallel_parameters_slurm_time_limit_min
        self.rfimpute.parameters.slurm_parameters.par_job_name                  = parameter_set.constants.parallel_slurm_parameters_par_job_name
        self.rfimpute.parameters.slurm_parameters.job_name                      = parameter_set.p_imputation_rfimpute_parallel_parameters_slurm_job_name
        self.rfimpute.parameters.slurm_parameters.par_output                    = parameter_set.constants.parallel_slurm_parameters_par_output
        self.rfimpute.parameters.slurm_parameters.output_ext                    = parameter_set.p_imputation_rfimpute_parallel_parameters_slurm_output_ext
        self.rfimpute.parameters.slurm_parameters.par_error                     = parameter_set.constants.parallel_slurm_parameters_par_error
        self.rfimpute.parameters.slurm_parameters.error_ext                     = parameter_set.p_imputation_rfimpute_parallel_parameters_slurm_error_ext
        
        self.rfimpute.parameters.slurm_parameters.script_path                   = parameter_set.p_imputation_rfimpute_parallel_parameters_slurm_script_path
        self.rfimpute.parameters.slurm_parameters.shell_script_path             = parameter_set.p_imputation_rfimpute_parallel_parameters_slurm_shell_script_path

        
        
        
    def impute_data_matrix(self):
        t_compendium_collections = self.get_t_compendium_collections()
        #Parameter configuration:
        #self.rfimpute.parameters.parallel_options = rfimpute.ParallelOptions.LOCAL.value #FOR TESTING
        
        merged_data_matrix = t_compendium_collections.get_merged_data_matrix()

        if self.parameters.skip_imputation == False or self.check_existed_imputation_results() == False:
            if self.parameters.imputation_options == ImputationOptions.RF.value:
                imputed_data_matrix = np.transpose(self.rfimpute.miss_forest_imputation(np.transpose(merged_data_matrix)))
            
            imputed_data_matrix = pd.DataFrame(data = imputed_data_matrix, index = merged_data_matrix.index, columns = merged_data_matrix.columns)
            
        else:
            imputed_data_matrix = pd.read_csv(self.parameters.imputed_data_matrix_path,index_col = 0)

        
        t_compendium_collections.set_imputed_data_matrix(imputed_data_matrix, 
                                                        self.parameters.imputation_options,
                                                        self.parameters.imputed_data_matrix_path)
        
        t_compendium_collections.output_imputed_data_matrix()
    def check_existed_imputation_results(self):
        if not os.path.isfile(self.parameters.imputed_data_matrix_path):
            return False
        else:
            return True
        
        
    
        
