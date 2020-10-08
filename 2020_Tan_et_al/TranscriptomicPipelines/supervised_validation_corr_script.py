from transcriptomic_pipeline import *
import pickle
import sys
import os
import os.path
if __name__ == '__main__':
    try:
        project_name = sys.argv[1]
        input_corr_path = sys.argv[2]
    except:
        raise Exception('Usage: python supervised_validation_corr_script.py <project name> <study-condition-sample_mapping_table>')
        
    if not os.path.isdir(project_name):
        raise Exception('Error: project ' + project_name + ' not found!')
        
    os.chdir(project_name)
    input_corr_path = '../' + input_corr_path
    if not os.path.isfile(project_name+'_projectfile.bin'):
        raise Exception('Error: the compendium is not exist! Please generate the compendium first!')
        
    if not os.path.isfile(input_corr_path):
        raise Exception('Error: the study-condition-sample_mapping_table does not exist!')
    
    transcriptomic_pipeline = pickle.load(open(project_name+'_projectfile.bin','rb'))
    transcriptomic_pipeline.parameter_set.v_supervised_parameters_correlation_validation_results_path = project_name + '_CorrelationValidationResults.csv'
    transcriptomic_pipeline.parameter_set.v_supervised_parameters_correlation_validation_results_figure_path = project_name + '_CorrelationValidationResults.png'
    
    transcriptomic_pipeline.validation_pipeline.configure_parameter_set_all()
    transcriptomic_pipeline.validation_pipeline.supervised_validation.correlation_validation(input_corr_path)