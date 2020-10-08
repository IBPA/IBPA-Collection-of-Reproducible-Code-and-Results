from transcriptomic_pipeline import *
import pickle
import sys
import os
import os.path
if __name__ == '__main__':
    try:
        project_name = sys.argv[1]
        published_data_path = sys.argv[2]
    except:
        raise Exception('Usage: python supervised_validation_published_data_comparison_script.py <project name> <published_data_path>')
        
    if not os.path.isdir(project_name):
        raise Exception('Error: project ' + project_name + ' not found!')
        
    os.chdir(project_name)
    published_data_path = '../' + published_data_path
    if not os.path.isfile(project_name+'_projectfile.bin'):
        raise Exception('Error: the compendium is not exist! Please generate the compendium first!')
        
    if not os.path.isfile(published_data_path):
        raise Exception('Error: the published data does not exist!')
    
    transcriptomic_pipeline = pickle.load(open(project_name+'_projectfile.bin','rb'))
    transcriptomic_pipeline.parameter_set.v_supervised_parameters_correlation_validation_results_path = project_name + '_PublishedDataComparisonResults.csv'
    transcriptomic_pipeline.parameter_set.v_supervised_parameters_correlation_validation_results_figure_path = project_name + '_PublishedDataComparisonResults.png'
    
    transcriptomic_pipeline.validation_pipeline.configure_parameter_set_all()
    transcriptomic_pipeline.validation_pipeline.supervised_validation.published_data_comparison_range_capture(published_data_path)