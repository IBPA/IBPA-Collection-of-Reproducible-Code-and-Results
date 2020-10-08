import os
import os.path
import sys

def print_usage(): 
        print('Usage: python clean_intermediate_data.py <project_name> <data option>')
        print('Data Option:')
        print('    sample_table      ==> for adding/remove sample from compendium')
        print('    gene_count        ==> for changing parameters of value extraction')
        print('                       (NOTE: It will also remove sample mapping results and postprocessing results')
        print('    sample_mapping    ==> for changing parameters of sample mapping')
        print('                       (NOTE: It will also remove postprocessing results')
        print('    postprocessing    ==> for changing parameters of postprocessing (including normalization)')
        print('    normalized_matrix ==> for changing parameters of normalization (normalization only')
        print('After cleaning the intermediate data, the compendium will be removed and you have to run the build_compendium to build the compendium again')

def remove_gene_count():
    files = os.listdir('.')
    for cur_file in files:
        if cur_file.startswith('results_s_value_extraction') and cur_file.endswith('.json'):
            os.remove(cur_file)

def remove_sample_mapping():
    files = os.listdir('.')
    for cur_file in files:
        if cur_file.startswith('results_s_sample_mapping') and cur_file.endswith('.json'):
            os.remove(cur_file)

def remove_post_processing(project_name):
    MergedDataMatrixPath = project_name + '_MergedDataMatrix.csv'
    MergedMetadataTablePath = project_name + '_MergedMetadatatable.csv'
    ImputedDataMatrixPath = project_name + '_InputedDataMatrix.csv'
    
    if os.path.isfile(MergedDataMatrixPath):
        os.remove(MergedDataMatrixPath)
    if os.path.isfile(MergedMetadataTablePath):
        os.remove(MergedMetadataTablePath)
    if os.path.isfile(ImputedDataMatrixPath):
        os.remove(ImputedDataMatrixPath)

    remove_normalized_matrix(project_name)

def remove_normalized_matrix(project_name):
    NormalizedDataMatrixPath = project_name + '_NormalizedDataMatrix.csv'
    if os.path.isfile(NormalizedDataMatrixPath):
        os.remove(NormalizedDataMatrixPath)

def remove_compendium(project_name):
    if os.path.isfile(project_name + '_projectfile.bin'):
        os.remove(project_name + '_projectfile.bin')

if __name__ == '__main__':
    try:
        project_name = sys.argv[1]
        clean_target = sys.argv[2]
    except:
        print_usage()
        raise Exception


    if not os.path.isdir(project_name):
        raise Exception('The project ' + project_name + ' is not found!')

    os.chdir(project_name)
    if clean_target == 'sample_table':
        try:
            os.remove(project_name+'_sra_run_info.csv')
        except:
            raise Exception('The sample table does not exist.')

    elif clean_target == 'gene_count':
        print('Warning: This will remove all sample value extraction results.')
        print('Extract all sample value extraction results may be time consuming!')
        print('Do you want to continue? [yes/NO]')
        answer = str(input())
        if answer != 'Yes' and answer != 'yes':
            print('Please type yes to confirm')
            quit()

        remove_gene_count()
        remove_sample_mapping()
        remove_post_processing(project_name)
        remove_compendium(project_name)

    elif clean_target == 'sample_mapping':
        remove_sample_mapping()
        remove_post_processing(project_name)
        remove_compendium(project_name)

    elif clean_target == 'postprocessing':
        remove_post_processing(project_name)
        remove_compendium(project_name)

    elif clean_target == 'normalized_matrix':
        remove_normalized_matrix(project_name)
        remove_compendium(project_name)
