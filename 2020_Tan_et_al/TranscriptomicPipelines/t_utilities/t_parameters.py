import numpy as np
#User-side Parameters Configurations
class TranscriptomicConstants:
    #CONSTANT PARTS: SHOULD NOT BE MODIFIED UNLESS THE RELATED TOOLKITS ARE UPDATED/CHANGED!
    def __init__(self):
        #Parallel
        self.parallel_option_none                                                   = 'None'
        self.parallel_option_local                                                  = 'local'
        self.parallel_option_slurm                                                  = 'SLURM'
        
        #Parallel (SLURM)
        self.parallel_slurm_parameters_par_num_node                                 = '-N'
        self.parallel_slurm_parameters_par_num_core_each_node                       = '-c'
        self.parallel_slurm_parameters_par_time_limit                               = '--time'
        self.parallel_slurm_parameters_par_job_name                                 = '-J'
        self.parallel_slurm_parameters_par_output                                   = '-o'
        self.parallel_slurm_parameters_par_error                                    = '-e'
        
        #sequencing
        #sequencing_general: None in this version
        #
        #
        #
        #
        #related toolkits: 
        
        #   SRAToolkit (For SRA data processing)=====================================
        #       SRATool -- Prefetch
        self.s_sratool_parameters_prefetch_par_output_file                          = '-o'
        self.s_sratool_parameters_prefetch_par_force                                = '-f'
        self.s_sratool_parameters_prefetch_par_output_dir                           = '-O'
        
        #       SRATool -- Validate
        #(EMPTY)
        
        #       SRATool -- FastqDump
        self.s_sratool_parameters_fastqdump_par_gzip                                = '--gzip'
        self.s_sratool_parameters_fastqdump_par_split3                              = '--split-e'
        self.s_sratool_parameters_fastqdump_par_output_dir                          = '-O'

        #   Bowtie2 (For alignment)==================================================
        #       Bowtie2 -- index building
        self.s_bowtie2_parameters_build_par_nthreads                                = '--threads'
        
        #       Bowtie2 -- alignment
        #           Bowtie2 -- alignment -- Input and Output file configuration
        self.s_bowtie2_parameters_align_par_index_name                              = '-x'
        self.s_bowtie2_parameters_align_par_paired_1                                = '-1'
        self.s_bowtie2_parameters_align_par_paired_2                                = '-2'
        self.s_bowtie2_parameters_align_par_unpaired                                = '-U'
        self.s_bowtie2_parameters_align_par_sam                                     = '-S'
        self.s_bowtie2_parameters_align_par_fastq                                   = '-q'
        self.s_bowtie2_parameters_align_par_phred33                                 = '--phred33'
        self.s_bowtie2_parameters_align_par_phred64                                 = '--phred64'
        
        #           Bowtie2 -- alignment -- Sensitivity and Speed Mode
        self.s_bowtie2_parameters_align_par_very_fast                               = '--very-fast'
        self.s_bowtie2_parameters_align_par_fast                                    = '--fast'
        self.s_bowtie2_parameters_align_par_sensitive                               = '--sensitive'
        self.s_bowtie2_parameters_align_par_very_sensitive                          = '--very-sensitive'
        self.s_bowtie2_parameters_align_par_very_fast_local                         = '--very-fast-local'
        self.s_bowtie2_parameters_align_par_fast_local                              = '--fast-local'
        self.s_bowtie2_parameters_align_par_sensitive_local                         = '--sensitive-local'
        self.s_bowtie2_parameters_align_par_very_sensitive_local                    = '--very-sensitive-local'
        
        #           Bowtie2 -- alignment -- Local or Global Mode
        self.s_bowtie2_parameters_align_par_end_to_end                              = '--end-to-end'
        self.s_bowtie2_parameters_align_par_local                                   = '--local'
        
        #           Bowtie2 -- alignment -- Parallel Configurations
        self.s_bowtie2_parameters_align_par_nthreads                                = '--threads'
        self.s_bowtie2_parameters_align_par_reorder                                 = '--reorder'
        self.s_bowtie2_parameters_align_par_mm                                      = '--mm'
        
        
        #   RSeQC (For stranded/unstranded detection)================================
        #           RSeQC   -- infer_experiment
        self.s_rseqc_parameters_infer_experiment_par_bed                            = '-r'
        self.s_rseqc_parameters_infer_experiment_par_input                          = '-i'
        
        
        #   HTSeq (For Read Counts)==================================================
        self.s_htseq_parameters_dir                                                 = ''
        
        #           HTSeq   -- htseq-count
        self.s_htseq_parameters_htseq_count_par_target_type                         = '-t'
        self.s_htseq_parameters_htseq_count_par_used_id                             = '-i'
        self.s_htseq_parameters_htseq_count_par_quiet                               = '-q'
        self.s_htseq_parameters_htseq_count_par_stranded                            = '-s'
        
        #
        #
        #
        #
        #Related Modules
        
        #   Sample Mapping
        self.s_sample_mapping_parameters_different_run_merge_mode_option_drop_experiment    = 'drop'
        self.s_sample_mapping_parameters_different_run_merge_mode_option_random_run         = 'random'
        self.s_sample_mapping_parameters_different_run_merge_mode_option_specified_run      = 'specified'
        self.s_sample_mapping_parameters_different_run_merge_mode_option_average            = 'average'
        
        
        #PostProcessing
        #
        #
        #
        #
        #Validation
        self.v_unsupervised_parameters_impute_mode_randomforest                             = 'random_forest'
        self.v_unsupervised_parameters_impute_mode_knn                                      = 'knn'
        
        
        
        #Related Modules:
        #   Imputation
        self.p_imputation_parameters_imputation_options_average                             = 'average'
        self.p_imputation_parameters_imputation_options_zero                                = 'zero'
        self.p_imputation_parameters_imputation_options_knn                                 = 'knn'
        self.p_imputation_parameters_imputation_options_rf                                  = 'random_forest'
        
        self.p_imputation_rfimpute_parameters_initial_guess_options_average                 = 'average'
        self.p_imputation_rfimpute_parameters_initial_guess_options_zero                    = 'zero'
        self.p_imputation_rfimpute_parameters_initial_guess_options_knn                     = 'knn'
        
        #   Normalization
        self.p_normalization_parameters_normalization_options_quantile                      = 'quantile'
        self.p_normalization_parameters_normalization_options_combat                        = 'combat'
        self.p_normalization_parameters_normalization_options_none                          = 'none'
        
class TranscriptomicParameters:
    def __init__(self, owner):
        self.owner = owner
        self.constants = TranscriptomicConstants()
        
        #general_parameters
        self.general_parameters_executive_prefix                                    = ""
        self.general_parameters_executive_surfix                                    = ""
        self.general_parameters_dir_sep                                             = ""
        
        #sequencing
        #sequencing_general: None in this version
        #
        #
        #
        #
        #Related toolkits: 
        
        #   SRAToolkit (For SRA data processing)=====================================
        self.s_sratool_parameters_dir                                               = ''
        
        #       SRATool -- Prefetch
        self.s_sratool_parameters_prefetch_exe_file                                 = 'prefetch'
        self.s_sratool_parameters_prefetch_force                                    = 'all'
        
        #       SRATool -- Validate
        self.s_sratool_parameters_validate_exe_file                                 = 'vdb-validate'
        
        #       SRATool -- FastqDump
        self.s_sratool_parameters_fastqdump_exe_file                                = 'fastq-dump'
        
        
        #   Bowtie2 (For alignment)==================================================
        self.s_bowtie2_parameters_dir                                               = ''
        
        #       Bowtie2 -- index building
        self.s_bowtie2_parameters_build_exe_file                                    = 'bowtie2-build'
        self.s_bowtie2_parameters_build_nthreads                                    = 31
        #self.s_bowtie2_parameters_build_index_name                                  = 'TestTemplate'
        
        #       Bowtie2 -- alignment
        self.s_bowtie2_parameters_align_exe_file                                    = 'bowtie2'
        
        #           Bowtie2 -- alignment -- Input and Output file configuration
        #(EMPTY)
        
        #           Bowtie2 -- alignment -- Sensitivity and Speed Mode
        self.s_bowtie2_parameters_align_speed                                       = self.constants.s_bowtie2_parameters_align_par_sensitive
        
        #           Bowtie2 -- alignment -- Local or Global Mode
        self.s_bowtie2_parameters_align_mode                                        = self.constants.s_bowtie2_parameters_align_par_end_to_end
        
        #           Bowtie2 -- alignment -- Parallel Configurations
        self.s_bowtie2_parameters_align_nthreads                                    = 4
        
        
        #   RSeQC (For stranded/unstranded detection)================================
        self.s_rseqc_parameters_dir                                                 = ''
        
        #           RSeQC   -- infer_experiment
        self.s_rseqc_parameters_infer_experiment_exe_file                           = 'infer_experiment.py'
        
        
        #   HTSeq (For Read Counts)==================================================
        self.s_htseq_parameters_dir                                                 = ''
        
        #           HTSeq   -- htseq-count
        self.s_htseq_parameters_htseq_count_exe_file                                = 'htseq-count'

        
        #
        #
        #
        #
        #Related Modules
        #Data Retrieval==============================================================
        self.s_data_retrieval_parameters_entrez_mail                                = 'cetan@ucdavis.edu'
        self.s_data_retrieval_parameters_sra_run_info_path                          = 'sra_run_info.csv'
        self.s_data_retrieval_parameters_fasta_path                                 = 'merged.fasta'
        self.s_data_retrieval_parameters_skip_srainfo_download                      = True
        self.s_data_retrieval_parameters_skip_fasta_download                        = True
        
        
        #Value Extraction============================================================
        self.s_value_extraction_parameters_working_file_dir                         = '.'
        self.s_value_extraction_parameters_alignment_record_file_ext                = '.align_out'
        self.s_value_extraction_parameters_infer_experiment_record_file_ext         = '.infer_experiment_out'
        self.s_value_extraction_parameters_infer_experiment_threshold               = 0.9
        self.s_value_extraction_parameters_count_reads_file_ext                     = '.count_reads'
        self.s_value_extraction_parameters_n_trial                                  = 1
        self.s_value_extraction_parameters_skip_all                                 = True
        self.s_value_extraction_parameters_skip_fastq_dump                          = True
        self.s_value_extraction_parameters_skip_alignment                           = True
        self.s_value_extraction_parameters_skip_infer_experiment                    = True
        self.s_value_extraction_parameters_skip_count_reads                         = True
        self.s_value_extraction_parameters_clean_reference_genome                   = False
        self.s_value_extraction_parameters_clean_existed_sra_files                  = True
        self.s_value_extraction_parameters_clean_existed_fastqdump_results          = True
        self.s_value_extraction_parameters_clean_existed_alignment_sequence_results = True
        self.s_value_extraction_parameters_clean_existed_alignment_results          = True
        self.s_value_extraction_parameters_clean_existed_infer_experiment_results   = True
        self.s_value_extraction_parameters_clean_existed_count_read_results         = True
        self.s_value_extraction_parameters_clean_existed_worker_file                = True
        self.s_value_extraction_parameters_clean_existed_results                    = True
        
        #Value Extraction (Reference Genome Building)
        self.s_value_extraction_refbuild_parameters_parallel_mode                    = self.constants.parallel_option_slurm
        self.s_value_extraction_refbuild_parameters_n_processes_local               = 31
        self.s_value_extraction_refbuild_parameters_n_jobs_slurm                    = 1
        self.s_value_extraction_refbuild_parameters_slurm_num_core_each_node        = 30
        self.s_value_extraction_refbuild_parameters_slurm_time_limit_hr             = 10
        self.s_value_extraction_refbuild_parameters_slurm_time_limit_min            = 0
        self.s_value_extraction_refbuild_parameters_slurm_output_ext                = '.output'
        self.s_value_extraction_refbuild_parameters_slurm_error_ext                 = '.error'
        self.s_value_extraction_refbuild_parameters_slurm_shell_script_path         = 'job.sh'
        self.s_value_extraction_refbuild_parameters_skip_build                        = True
        
        #Value Extraction (Parallel)=================================================
        self.s_value_extraction_parallel_parameters_pyscript                        = 'script_get_read_counts_run.py'
        self.s_value_extraction_parallel_parameters_parallel_mode                   = self.constants.parallel_option_slurm
        self.s_value_extraction_parallel_parameters_n_processes_local               = 2
        self.s_value_extraction_parallel_parameters_n_jobs_slurm                    = 6
        self.s_value_extraction_parallel_parameters_slurm_num_core_each_node        = 4
        self.s_value_extraction_parallel_parameters_slurm_time_limit_hr             = 16
        self.s_value_extraction_parallel_parameters_slurm_time_limit_min            = 0
        self.s_value_extraction_parallel_parameters_slurm_output_ext                = '.output'
        self.s_value_extraction_parallel_parameters_slurm_error_ext                 = '.error'
        self.s_value_extraction_parallel_parameters_slurm_shell_script_path         = 'job.sh'
        
        #Sample Mapping
        self.s_sample_mapping_parameters_different_run_merge_mode                   = self.constants.s_sample_mapping_parameters_different_run_merge_mode_option_average
        self.s_sample_mapping_parameters_n_trial                                    = 10
        self.s_sample_mapping_parameters_skip_merge_different_run                   = True
        self.s_sample_mapping_parameters_clean_existed_worker_file                  = True
        self.s_sample_mapping_parameters_clean_existed_results                      = True
        
        #Sample Mapping (Parallel)=================================================
        self.s_sample_mapping_parallel_parameters_pyscript                          = '../script_merge_runs.py'
        self.s_sample_mapping_parallel_parameters_parallel_mode                     = self.constants.parallel_option_slurm
        self.s_sample_mapping_parallel_parameters_n_processes_local                 = 2
        self.s_sample_mapping_parallel_parameters_n_jobs_slurm                      = 4
        self.s_sample_mapping_parallel_parameters_slurm_num_core_each_node          = 30
        self.s_sample_mapping_parallel_parameters_slurm_time_limit_hr               = 1
        self.s_sample_mapping_parallel_parameters_slurm_time_limit_min              = 0
        self.s_sample_mapping_parallel_parameters_slurm_output_ext                  = '.output'
        self.s_sample_mapping_parallel_parameters_slurm_error_ext                   = '.error'
        self.s_sample_mapping_parallel_parameters_slurm_shell_script_path           = 'job.sh'
        
        #Gene Mapping
        self.s_gene_mapping_parameters_drop_unnamed_genes                           = False
        self.s_gene_mapping_parameters_use_gene_names                               = False
        self.s_gene_mapping_parameters_data_matrix_table_path                       = 'SequencingDataMatrix.csv'
        self.s_gene_mapping_parameters_gene_mapping_table_path                      = 'SequencingGeneMappingTable.csv'
        
        
        #PostProcessing
        #
        #
        #
        #
        #Related Modules:
        #   Concatenation
        self.p_concatenation_parameters_merged_metadata_table_path                  = 'MergedMetadatatable.csv'
        self.p_concatenation_parameters_merged_data_matrix_path                     = 'MergedDataMatrix.csv'
        
        #   Imputation
        self.p_imputation_parameters_imputation_option                              = self.constants.p_imputation_parameters_imputation_options_rf
        self.p_imputation_parameters_imputed_data_matrix_path                       = "ImputedDataMatrix.csv"
        self.p_imputation_parameters_skip_imputation                                = True

        
        #   Imputation (RFImpute)
        self.p_imputation_rfimpute_parameters_initial_guess_option                  = self.constants.p_imputation_parameters_imputation_options_average
        self.p_imputation_rfimpute_parameters_max_iter                              = 10
        #   Imputation (RFImpute -- Parallel) (Should be used in the future)
        self.p_imputation_rfimpute_parallel_parameters_parallel_mode                = self.constants.parallel_option_slurm
        self.p_imputation_rfimpute_parallel_parameters_n_feature_local              = 200
        self.p_imputation_rfimpute_parallel_parameters_n_jobs                       = 1
        self.p_imputation_rfimpute_parallel_parameters_n_core_local                 = 32
        self.p_imputation_rfimpute_parallel_parameters_slurm_time_limit_hr          = 1
        self.p_imputation_rfimpute_parallel_parameters_slurm_time_limit_min         = 0
        self.p_imputation_rfimpute_parallel_parameters_slurm_job_name               = 'RFImputation'
        self.p_imputation_rfimpute_parallel_parameters_slurm_output_ext             = '.output'
        self.p_imputation_rfimpute_parallel_parameters_slurm_error_ext              = '.error'
        self.p_imputation_rfimpute_parallel_parameters_slurm_script_path            = '../utilities/imputation/job.py'
        self.p_imputation_rfimpute_parallel_parameters_slurm_shell_script_path      = 'job.sh'
        self.p_imputation_rfimpute_parallel_parameters_slurm_tmp_X_file             = 'tmp_X.dat'
        
        #   Normalization
        self.p_normalization_parameters_normalization_option                        = self.constants.p_normalization_parameters_normalization_options_quantile
        self.p_normalization_parameters_normalized_data_matrix_path                 = 'NormalizedDataMatrix.csv'
        
        
        #Validation
        #
        #
        #
        #
        #Related Modules:
        #   Unsupervised
        self.v_unsupervised_parameters_n_trial                                      = 1
        self.v_unsupervised_parameters_noise_ratio                                  = np.arange(0,1.1,0.2)
        self.v_unsupervised_parameters_missing_value_ratio                          = np.append(np.arange(0.3,0.9,0.2),0.99)
        self.v_unsupervised_parameters_results_path                                 = 'UnsupervisedValidationResults.csv'
        self.v_unsupervised_parameters_results_figure_path                          = 'UnsupervisedValidationResults.png'
        self.v_unsupervised_parameters_skip_validate_data                           = False
        self.v_unsupervised_parameters_impute_mode                                  = self.constants.v_unsupervised_parameters_impute_mode_knn
        #   (Share the imputation parameters)
        
        
        self.v_supervised_parameters_n_trial                                        = 10
        self.v_supervised_parameters_noise_ratio                                    = np.arange(0,1.1,0.1)
        self.v_supervised_parameters_correlation_validation_results_path            = 'CorrelationValidationResults.csv'
        self.v_supervised_parameters_knowledge_capture_validation_results_path      = 'KnowledgeCaptureValidationResults.csv'
        self.v_supervised_parameters_published_data_comparison_results_path                = 'PublishedDataComparisonResults.csv'
        self.v_supervised_parameters_correlation_validation_results_figure_path            = 'CorrelationValidationResults.png'
        self.v_supervised_parameters_knowledge_capture_validation_results_figure_path      = 'KnowledgeCaptureValidationResults.png'
        self.v_supervised_parameters_published_data_comparison_results_figure_path         = 'PublishedDataComparisonResults.png'
        
        
