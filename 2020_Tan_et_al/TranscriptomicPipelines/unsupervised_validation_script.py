from transcriptomic_pipeline import *
import pickle
import sys
import os
import os.path
if __name__ == '__main__':
    try:
        project_name = sys.argv[1]
    except:
        raise Exception('Usage: python unsupervised_validation_script.py <project name>')
        
    if not os.path.isdir(project_name):
        raise Exception('Error: project ' + project_name + ' not found!')
        
    os.chdir(project_name)
    if not os.path.isfile(project_name+'_projectfile.bin'):
        raise Exception('Error: the compendium is not exist! Please generate the compendium first!')
    
    transcriptomic_pipeline = pickle.load(open(project_name+'_projectfile.bin','rb'))
    transcriptomic_pipeline.parameter_set.v_unsupervised_parameters_results_path = project_name + '_UnsupervisedValidationResults.csv'
    transcriptomic_pipeline.parameter_set.v_unsupervised_parameters_results_figure_path = project_name + '_UnsupervisedValidationResults.png'
    
    transcriptomic_pipeline.validation_pipeline.configure_parameter_set_all()
    transcriptomic_pipeline.validation_pipeline.unsupervised_validation.validate_data()