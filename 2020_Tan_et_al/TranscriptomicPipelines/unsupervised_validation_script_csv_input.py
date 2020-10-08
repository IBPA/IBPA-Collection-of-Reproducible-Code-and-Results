from transcriptomic_pipeline import *
import pickle
import sys
import os
import os.path

import pandas as pd
if __name__ == '__main__':
    try:
        project_name = sys.argv[1]
        data_matrix_csv = sys.argv[2]
        reference_output_csv = sys.argv[3]
        reference_output_figure = sys.argv[4]
    except:
        raise Exception('Usage: python unsupervised_validation_script_csv_input.py <project name> <data_matrix> <output_table_path> <output_figure_path>')
        
    if not os.path.isdir(project_name):
        raise Exception('Error: project ' + project_name + ' not found!')
        
    os.chdir(project_name)
    data_matrix_csv = '../' + data_matrix_csv
    if not os.path.isfile(project_name+'_projectfile.bin'):
        raise Exception('Error: the compendium is not exist! Please generate the compendium first!')
    
    transcriptomic_pipeline = pickle.load(open(project_name+'_projectfile.bin','rb'))

    transcriptomic_pipeline.validation_pipeline.configure_parameter_set_all()
    
    data_matrix = pd.read_csv(data_matrix_csv, index_col = 0)
    
    transcriptomic_pipeline.validation_pipeline.unsupervised_validation.validate_data_from_data_matrix(data_matrix, reference_output_csv, reference_output_figure)