import sys
if (sys.version_info < (3, 0)):
    import v_module_template
else:
    from . import v_module_template
    
import numpy as np
import pandas as pd
from missingpy import KNNImputer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import os
import time


#RF Impute
#NOTE: Later this part should become a independent package so that we should import it directly
sys.path.insert(0,"../utilities/imputation")
import rfimpute
    
class UnsupervisedValidationParameters:
    def __init__(   self, n_trial = 2,
                    noise_ratio = np.arange(0,1.1,0.1),
                    missing_value_ratio = np.arange(0.3,0.8,0.2),
                    unsupervised_validation_results_path = "UnsupervisedValidationResults.csv",
                    unsupervised_validation_results_figure_path = "UnsupervisedValidationResults.png",
                    impute_mode = 'knn'):
        self.n_trial = n_trial
        self.noise_ratio = noise_ratio
        self.missing_value_ratio = missing_value_ratio
        self.unsupervised_validation_results_path = unsupervised_validation_results_path
        self.unsupervised_validation_results_figure_path = unsupervised_validation_results_figure_path
        self.impute_mode = impute_mode
        
        self.skip_validate_data = True
       
    def get_unsupervised_validation_results_path(self):
        return self.unsupervised_validation_results_path
    
    
class UnsupervisedValidation(v_module_template.ValidationSubModule):
    def __init__(self, owner):
        self.owner = owner
        self.rfimpute = rfimpute.MissForestImputation()
        self.parameters = UnsupervisedValidationParameters()
        
        self.configure_parameter_set()
        self.configure_rfimpute_parameter_set()
        
    def configure_parameter_set(self):
        parameter_set = self.get_parameter_set()
        self.parameters.n_trial                                         = parameter_set.v_unsupervised_parameters_n_trial
        self.parameters.noise_ratio                                     = parameter_set.v_unsupervised_parameters_noise_ratio
        self.parameters.missing_value_ratio                             = parameter_set.v_unsupervised_parameters_missing_value_ratio
        self.parameters.unsupervised_validation_results_path            = parameter_set.v_unsupervised_parameters_results_path
        self.parameters.unsupervised_validation_results_figure_path     = parameter_set.v_unsupervised_parameters_results_figure_path
        self.parameters.impute_mode                                     = parameter_set.v_unsupervised_parameters_impute_mode
        self.parameters.skip_validate_data                              = parameter_set.v_unsupervised_parameters_skip_validate_data
        
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
        self.rfimpute.parameters.slurm_parameters.num_node                      = 1 
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
        
    def validate_data(self):
        if self.parameters.skip_validate_data == False or self.check_existed_result() == False:
        #if False:
            #self.rfimpute.parameters.parallel_options = rfimpute.ParallelOptions.LOCAL.value #FOR TESTING
            
            t_compendium_collections = self.get_t_compendium_collections()
            normalized_data_matrix = t_compendium_collections.get_normalized_data_matrix()
            
            self.validate_data_from_data_matrix(normalized_data_matrix)
            '''
            
            normalized_data_matrix_nparray = np.array(normalized_data_matrix)
            #scaler = MinMaxScaler()
            #scaler.fit(normalized_data_matrix_nparray)
            #minmax_normalized_data_matrix_nparray = scaler.transform(normalized_data_matrix_nparray)
            
            results = np.zeros((len(self.parameters.noise_ratio), len(self.parameters.missing_value_ratio)))
            
            for i in range(self.parameters.n_trial):
                print("Trial : " + str(i))
                cur_results = np.zeros((len(self.parameters.noise_ratio), len(self.parameters.missing_value_ratio)))
                noise = self.get_noise(normalized_data_matrix_nparray)
                
                for j in range(len(self.parameters.noise_ratio)):
                    noise_ratio = self.parameters.noise_ratio[j]
                    cur_matrix = noise*noise_ratio + normalized_data_matrix_nparray*(1-noise_ratio)
                    for k in range(len(self.parameters.missing_value_ratio)):
                        print("k: " + str(k) + "/" + str(len(self.parameters.missing_value_ratio)))
                        missing_value_ratio = self.parameters.missing_value_ratio[k]
                        missing_value_index = []
                        cur_matrix_missing = np.copy(cur_matrix)
                        n_missing_value_each_col = np.round(missing_value_ratio*cur_matrix.shape[0]).astype(int)
                        start = time.time()
                        for colidx in range(cur_matrix.shape[1]):
                            cur_misi = np.random.choice(cur_matrix.shape[0],n_missing_value_each_col,replace = False)
                            cur_matrix_missing[cur_misi,colidx] = np.nan
                            missing_value_index.append(cur_misi)
                        end = time.time()
                        print("Generating missing values:" + str(end - start))
                        
                        start = time.time()
                        print("Do Impute!")
                        imputed_cur_matrix = self.do_impute(cur_matrix_missing)
                        print("End Impute!")
                        end = time.time()
                        print("Imputation:" + str(end - start))
                        
                        start = time.time()
                        correct_values = []
                        imputed_values = []
                        for colidx in range(cur_matrix.shape[1]):
                            correct_values.extend(cur_matrix[missing_value_index[colidx],colidx].tolist())
                            imputed_values.extend(imputed_cur_matrix[missing_value_index[colidx],colidx].tolist())
                        
                        imputed_values = np.array(imputed_values)+1e-6
                        correct_values = np.array(correct_values)+1e-6
                        if j == 0 and k == 0:
                            np.savetxt('imputed_values_'+str(j)+'_'+str(k)+'.csv',imputed_values, delimiter=',')
                            np.savetxt('correct_values_'+str(j)+'_'+str(k)+'.csv',correct_values, delimiter=',')
                        cur_results[j,k] = np.mean(np.abs(imputed_values - correct_values+0.0)/(np.abs(imputed_values+0.0) + np.abs(correct_values+0.0)))*100
                        end = time.time()
                        print("Calculating Error:" + str(end - start))
                        
                results = results + cur_results
            
            results = results/self.parameters.n_trial
            
            results = pd.DataFrame(results, index = self.parameters.noise_ratio.tolist(), columns = self.parameters.missing_value_ratio.tolist())
            results.to_csv(self.parameters.unsupervised_validation_results_path)
            '''
            
        results = pd.read_csv(self.parameters.unsupervised_validation_results_path, index_col = 0)
        results.columns.name = 'missing value ratio'
        fig = plt.figure()
        unsupervised_plot = results.plot(title = 'Unsupervised validation results (average of ' + str(self.parameters.n_trial) + ' trials)')
        unsupervised_plot.set(xlabel='noise ratio',ylabel='Symmetric mean absolute percentage error(%)')
        plt.gca().axhline(y=100, color = 'red', linestyle='--')
        plt.savefig(self.parameters.unsupervised_validation_results_figure_path)
        plt.close(fig)
        
        benchmark = self.calculate_benchmark(results)
        print('Benchmark: the area between error curve and reference (100% Error for all noise ratio) is ' + str(np.round(benchmark,2)) + '(%)')
        
    def check_noise_ratio(self):
        self.parameters.noise_ratio = list(np.round(self.parameters.noise_ratio, 2))
        
        if max(self.parameters.noise_ratio) > 1 or min(self.parameters.noise_ratio) < 0:
            raise Exception('The noise ratio for unsupervised validation should be a number of sequence and each number should be in [0,1]')
            
        if 0 not in self.parameters.noise_ratio:
            self.parameters.noise_ratio = self.parameters.noise_ratio.append(0)
            print('Warning: Noise ratio should include 0. Added')
        if 1 not in self.parameters.noise_ratio:
            self.parameters.noise_ratio = self.parameters.noise_ratio.append(1)
            print('Warning: Noise ratio should include 1. Added')
            
        self.parameters.noise_ratio.sort()
        
    def check_missing_value_ratio(self):
        self.parameters.missing_value_ratio = list(np.round(self.parameters.missing_value_ratio, 2))

        if max(self.parameters.missing_value_ratio) >= 1 or min(self.parameters.missing_value_ratio) < 0:
            raise Exception('The missing value ratio for unsupervised validation should be a number of sequence and each number should be in [0,1)')
            
        if 0.5 not in self.parameters.missing_value_ratio:
            self.parameters.missing_value_ratio = self.parameters.missing_value_ratio.append(0.5)
            print('Warning: Noise ratio should include 0.5 (for estimating a benchmark). Added')
        
        self.parameters.missing_value_ratio.sort()
        
    def calculate_benchmark(self, result):
        if 0.5 not in result.columns and '0.5' not in result.columns:
            print(result.columns)
            raise Exception('Result of missing value ratio = 0.5 should be provided to calculate a benchmark!')
        
        noise_ratios = result.index.to_series()
        final_area = 0
        for i in range(result.shape[0]):
            if i > 0:
                final_area += (result['0.5'].iloc[i] + result['0.5'].iloc[i-1])*(noise_ratios.iloc[i] - noise_ratios.iloc[i-1])/2
                print(result['0.5'].iloc[i] + result['0.5'].iloc[i-1])
                print('---')
                print(noise_ratios.iloc[i] - noise_ratios.iloc[i-1])
                
        return 100 - final_area
        
    def validate_data_from_data_matrix(self, data_matrix, 
                                output_table_path = None, 
                                output_figure_path = None):
                                
        if not output_table_path:
            output_table_path = self.parameters.unsupervised_validation_results_path
            
        if not output_figure_path:
            output_figure_path = self.parameters.unsupervised_validation_results_figure_path
                                
        data_matrix_nparray = np.array(data_matrix)
        #scaler = MinMaxScaler()
        #scaler.fit(data_matrix_nparray)
        #minmax_data_matrix_nparray = scaler.transform(data_matrix_nparray)
        
        results = np.zeros((len(self.parameters.noise_ratio), len(self.parameters.missing_value_ratio)))
        
        self.check_missing_value_ratio()
        self.check_noise_ratio()
        
        
        
        for i in range(self.parameters.n_trial):
            print("Trial : " + str(i))
            cur_results = np.zeros((len(self.parameters.noise_ratio), len(self.parameters.missing_value_ratio)))
            noise = self.get_noise(data_matrix_nparray)
            
            for j in range(len(self.parameters.noise_ratio)):
                noise_ratio = self.parameters.noise_ratio[j]
                cur_matrix = noise*noise_ratio + data_matrix_nparray*(1-noise_ratio)
                for k in range(len(self.parameters.missing_value_ratio)):
                    print("k: " + str(k) + "/" + str(len(self.parameters.missing_value_ratio)))
                    missing_value_ratio = self.parameters.missing_value_ratio[k]
                    missing_value_index = []
                    cur_matrix_missing = np.copy(cur_matrix)
                    n_missing_value_each_col = np.round(missing_value_ratio*cur_matrix.shape[0]).astype(int)
                    start = time.time()
                    for colidx in range(cur_matrix.shape[1]):
                        cur_misi = np.random.choice(cur_matrix.shape[0],n_missing_value_each_col,replace = False)
                        cur_matrix_missing[cur_misi,colidx] = np.nan
                        missing_value_index.append(cur_misi)
                    end = time.time()
                    print("Generating missing values:" + str(end - start) + 'sec')
                    
                    start = time.time()
                    print("Do Impute!")
                    imputed_cur_matrix = self.do_impute(cur_matrix_missing)
                    print("End Impute!")
                    end = time.time()
                    print("Imputation:" + str(end - start))
                    
                    start = time.time()
                    correct_values = []
                    imputed_values = []
                    for colidx in range(cur_matrix.shape[1]):
                        correct_values.extend(cur_matrix[missing_value_index[colidx],colidx].tolist())
                        imputed_values.extend(imputed_cur_matrix[missing_value_index[colidx],colidx].tolist())
                    
                    imputed_values = np.array(imputed_values)+1e-6
                    correct_values = np.array(correct_values)+1e-6
                    if j == 0 and k == 0:
                        np.savetxt('imputed_values_'+str(j)+'_'+str(k)+'.csv',imputed_values, delimiter=',')
                        np.savetxt('correct_values_'+str(j)+'_'+str(k)+'.csv',correct_values, delimiter=',')
                    cur_results[j,k] = np.mean(np.abs(imputed_values - correct_values+0.0)/(np.abs(imputed_values+0.0) + np.abs(correct_values+0.0)))*100
                    end = time.time()
                    print("Calculating Error:" + str(end - start))
                    
            results = results + cur_results
        
        results = results/self.parameters.n_trial
        
        results = pd.DataFrame(results, index = self.parameters.noise_ratio.tolist(), columns = self.parameters.missing_value_ratio)
        results.to_csv(output_table_path)
        results = pd.read_csv(self.parameters.unsupervised_validation_results_path, index_col = 0)
        results.columns.name = 'missing value ratio'
        fig = plt.figure()
        unsupervised_plot = results.plot(title = 'Unsupervised validation results (average of ' + str(self.parameters.n_trial) + ' trials)')
        unsupervised_plot.set(xlabel='noise ratio',ylabel='Symmetric mean absolute percentage error(%)')
        plt.gca().axhline(y=100, color = 'red', linestyle='--')
        plt.savefig(output_figure_path)
        plt.close(fig)
        
        benchmark = self.calculate_benchmark(results)
        print('Benchmark: the area between error curve and reference (100% Error for all noise ratio) is ' + str(np.round(benchmark,2)) + '(%)')
        
        
    def check_existed_result(self):
        if not os.path.isfile(self.parameters.unsupervised_validation_results_path):
            return False
        else:
            return True
        
            
    def do_impute(self, matrix_to_impute):
        parameter_set = self.get_parameter_set()
        np.savetxt('test_cur_matrix_missing.csv',matrix_to_impute)
        if self.parameters.impute_mode == parameter_set.constants.v_unsupervised_parameters_impute_mode_randomforest:
            imputed_cur_matrix = np.transpose(self.rfimpute.miss_forest_imputation(np.transpose(matrix_to_impute)))
        elif self.parameters.impute_mode == parameter_set.constants.v_unsupervised_parameters_impute_mode_knn:
            imputer = KNNImputer(row_max_missing=1,col_max_missing=1)
            imputed_cur_matrix = np.transpose(imputer.fit_transform(np.transpose(matrix_to_impute)))
        
        return imputed_cur_matrix
            
