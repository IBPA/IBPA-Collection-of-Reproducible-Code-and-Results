from transcriptomic_pipeline import *
import pickle
import sys
import os
import os.path
if __name__ == '__main__':
    try:
        project_name = sys.argv[1]
        knowledge_capture_sample = sys.argv[2]
        knowledge_capture_gene = sys.argv[3]
    except:
        raise Exception('Usage: python supervised_validation_corr_script.py <project name> <Selected Sample List> <Selected Gene List>')
        
    if not os.path.isdir(project_name):
        raise Exception('Error: project ' + project_name + ' not found!')
        
    os.chdir(project_name)
    knowledge_capture_sample = '../' + knowledge_capture_sample
    knowledge_capture_gene = '../' + knowledge_capture_gene
    if not os.path.isfile(project_name+'_projectfile.bin'):
        raise Exception('Error: the compendium is not exist! Please generate the compendium first!')
        
    if not os.path.isfile(knowledge_capture_sample):
        raise Exception('Error: the sample list does not exist!')
        
    if not os.path.isfile(knowledge_capture_gene):
        raise Exception('Error: the gene list does not exist!')
    
    transcriptomic_pipeline = pickle.load(open(project_name+'_projectfile.bin','rb'))
    transcriptomic_pipeline.parameter_set.v_supervised_parameters_knowledge_capture_validation_results_path = project_name + '_KnowledgeCaptureValidationResults.csv'
    transcriptomic_pipeline.parameter_set.v_supervised_parameters_knowledge_capture_validation_results_figure_path = project_name + '_KnowledgeCaptureValidationResults.png'
    
    transcriptomic_pipeline.validation_pipeline.configure_parameter_set_all()
    transcriptomic_pipeline.validation_pipeline.supervised_validation.knowledge_capture_validation(knowledge_capture_sample, knowledge_capture_gene)
    