from transcriptomic_pipeline import *
import sys
import pandas as pd
import os
import os.path

if __name__ == "__main__":
    
    try:
        sample_list_file = sys.argv[1]
        gff_file = sys.argv[2]
        unique_id = sys.argv[3]
    except:
        raise Exception("Usage: python transcriptome_pipeline_20200304.py <sample_list_file> <gene_annotation_file(gff)> <project_name>")

    if not os.path.exists(unique_id):
        os.mkdir(unique_id)
    
    os.chdir(unique_id)
    tmp = pd.read_csv('../'+sample_list_file)
    exp_list = tmp["Experiment"].tolist()
    exp_list = list(set(exp_list))

    transcriptome_pipeline = TranscriptomicDataPreparationPipeline([],exp_list,['../' + gff_file])
    
    #Create unique id for each project
    transcriptome_pipeline.parameter_set.s_value_extraction_parallel_parameters_pyscript = '../script_get_read_counts_run.py'
    transcriptome_pipeline.parameter_set.s_sample_mapping_parallel_pyscript = '../script_merge_runs.py'
    transcriptome_pipeline.parameter_set.p_imputation_rfimpute_parallel_parameters_slurm_script_path = '../../utilities/imputation/job.py'

    
    transcriptome_pipeline.parallel_engine.parameters.parameters_SLURM.shell_script_path = unique_id + "_job.sh"
    #tmp_t_parameters = t_parameters.TranscriptomicParameters(transcriptome_pipeline)
    transcriptome_pipeline.parameter_set.s_data_retrieval_parameters_sra_run_info_path = unique_id + '_sra_run_info.csv'

    transcriptome_pipeline.parameter_set.s_gene_mapping_parameters_data_matrix_table_path = unique_id + '_SequencingDataMatrix.csv'
    transcriptome_pipeline.parameter_set.s_gene_mapping_parameters_gene_mapping_table_path = unique_id + '_SequencingGeneMappingTable.csv'
    
    transcriptome_pipeline.parameter_set.p_concatenation_parameters_merged_metadata_table_path = unique_id + '_MergedMetadatatable.csv'
    transcriptome_pipeline.parameter_set.p_concatenation_parameters_merged_data_matrix_path = unique_id + '_MergedDataMatrix.csv'
    
    transcriptome_pipeline.parameter_set.p_imputation_rfimpute_parallel_parameters_slurm_shell_script_path = unique_id + 'job.sh'
    transcriptome_pipeline.parameter_set.p_imputation_rfimpute_parallel_parameters_slurm_tmp_X_file = unique_id + 'tmp_X.dat'
    transcriptome_pipeline.parameter_set.p_imputation_parameters_imputed_data_matrix_path = unique_id + '_ImputedDataMatrix.csv'

    transcriptome_pipeline.parameter_set.p_normalization_parameters_normalized_data_matrix_path = unique_id + '_NormalizedDataMatrix.csv'
    
    #transcriptome_pipeline.parameter_set.v_unsupervised_parameters_results_path = unique_id + '_UnsupervisedValidationResults.csv'
    #transcriptome_pipeline.parameter_set.v_supervised_parameters_correlation_validation_results_path = unique_id + '_CorrelationValidationResults.csv'
    #transcriptome_pipeline.parameter_set.v_supervised_parameters_knowledge_capture_validation_results_path = unique_id + '_KnowledgeCaptureValidationResults.csv'
    #transcriptome_pipeline.parameter_set.v_unsupervised_parameters_results_figure_path = unique_id + '_UnsupervisedValidationResults.png'
    #transcriptome_pipeline.parameter_set.v_supervised_parameters_correlation_validation_results_figure_path = unique_id + '_CorrelationValidationResults.png'
    #transcriptome_pipeline.parameter_set.v_supervised_parameters_knowledge_capture_validation_results_figure_path = unique_id + '_KnowledgeCaptureValidationResults.png'
    
    
    transcriptome_pipeline.configure_parameter_set_all()
    
    #Start Working
    s_platform_id_remove = []
    s_series_id_remove = []
    s_experiment_id_remove = []
    s_run_id_remove = []
    
    transcriptome_pipeline.sequencing_pipeline.run_sequencing_pipeline(s_platform_id_remove,s_series_id_remove,s_experiment_id_remove,s_run_id_remove)
    transcriptome_pipeline.postprocessing_pipeline.run_postprocessing_pipeline()
    
    pickle.dump(transcriptome_pipeline, open(unique_id+'_projectfile.bin','wb'))
    
    #transcriptome_pipeline.validation_pipeline.run_validation_pipeline( input_corr_path = '../' + validate_corr_input, 
    #                                                                    input_knowledge_capture_groupping_path = '../' + validate_knowledge_capture_input_samples, 
    #                                                                    input_knowledge_capture_gene_list_path = '../' + validate_knowledge_capture_input_genes)

    os.chdir('..')
