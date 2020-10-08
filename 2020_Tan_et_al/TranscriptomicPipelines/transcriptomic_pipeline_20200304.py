import microarray_pipeline
import sequencing_pipeline
import postprocessing_pipeline
import validation_pipeline
import parallel_engine

import subprocess

import uuid

import sys
import os
if (sys.version_info < (3, 0)):
    sys.path.insert(0, "t_utilities")
    import t_compendium
    import t_gff
    import t_parameters
else:
    import t_utilities.t_compendium as t_compendium
    import t_utilities.t_gff as t_gff
    import t_utilities.t_parameters as t_parameters

from enum import Enum 
import sys
import pickle

class GeneralConstant(Enum):
    WINDOWS                     = "win32"
    WINDOWS_EXECUTIVE_PREFIX    = ""
    WINDOWS_EXECUTIVE_SURFIX    = ""
    WINDOWS_DIR_SEP             = "\\"
    
    UNIX_EXECUTIVE_PREFIX       = ""
    UNIX_EXECUTIVE_SURFIX       = ""
    UNIX_DIR_SEP                = "/"
    
    CODEC                       ="utf-8"
    
class GeneralParameters:
    def __init__(self):
        self.use_shell = False
        self.executive_prefix = ""
        self.executive_surfix = ""
        self.dir_sep = ""
        
    def check_os(self):
        if sys.platform == GeneralConstant.WINDOWS.value:
            self.executive_prefix = GeneralConstant.WINDOWS_EXECUTIVE_PREFIX.value
            self.executive_surfix = GeneralConstant.WINDOWS_EXECUTIVE_SURFIX.value
            self.dir_sep = GeneralConstant.WINDOWS_DIR_SEP.value
            self.use_shell = True
        else:
            self.executive_prefix = GeneralConstant.UNIX_EXECUTIVE_PREFIX.value
            self.executive_surfix = GeneralConstant.UNIX_EXECUTIVE_SURFIX.value
            self.dir_sep = GeneralConstant.UNIX_DIR_SEP.value
            self.use_shell = False


class TranscriptomicDataPreparationPipeline:
    def __init__(   self,
                    m_query_id,
                    s_query_id,
                    gff_path, #GFF3 Table Paths
                    owner = None
                    ):
                    
        #Initialize Parameter Set:
        self.parameter_set = t_parameters.TranscriptomicParameters(self)
                    
        #Initialize Environment:
        self.general_constant = GeneralConstant
        self.general_parameters = GeneralParameters() #Temp arrangement, it should be moved to upper layer (OmicsDataPreparationPipeline)
        self.general_parameters.check_os()
                    
        #Initialize the metadata table:
        #metadata will be initialized here and it will shared by ALL COMPONENT in this pipeline
        #Initialize the gene annotation table:
        self.t_gene_annotation = t_gff.GeneAnnotation(gff_path)
        print("read gff")
        print(gff_path)
        try:
            self.t_gene_annotation.read_file()
        except:
            self.t_gene_annotation = t_gff.GeneAnnotation(gff_path)
            print("Failed to use attribute 'gene' as ID: Use 'locus' as gene ID and also extract gene field")
            self.t_gene_annotation.target_type = t_gff.GFF3Type.GENE.value
            self.t_gene_annotation.used_id = t_gff.GFF3AttributesHeader.LOCUSTAG.value
            self.t_gene_annotation.gene_name_id = t_gff.GFF3AttributesHeader.NAME.value
            self.t_gene_annotation.read_file()
            
        print("read gff: done")
        self.t_compendium_collections = t_compendium.TranscriptomeCompendiumCollections()
        
        #Parallel Engine
        self.parallel_engine = parallel_engine.ParallelEngine()
        
        self.microarray_pipeline = microarray_pipeline.MicroarrayPipeline(self, m_query_id, t_compendium.TranscriptomeCompendium())
        self.sequencing_pipeline = sequencing_pipeline.SequencingPipeline(self, s_query_id, t_compendium.TranscriptomeCompendium(query_ids = s_query_id))
        self.postprocessing_pipeline = postprocessing_pipeline.PostprocessingPipeline(self)
        self.validation_pipeline = validation_pipeline.ValidationPipeline(self)
        
        #Configure Parameters
    
    def configure_parameter_set_all(self):
        self.configure_parameter_set()
        self.sequencing_pipeline.configure_parameter_set_all(self.general_constant, self.general_parameters)
        self.postprocessing_pipeline.configure_parameter_set_all()
        self.validation_pipeline.configure_parameter_set_all()
    
    def configure_parameter_set(self):
        self.general_parameters.executive_prefix    = self.parameter_set.general_parameters_executive_prefix
        self.general_parameters.executive_surfix    = self.parameter_set.general_parameters_executive_surfix
        self.general_parameters.dir_sep             = self.parameter_set.general_parameters_dir_sep
        self.general_parameters.check_os()
        
    def get_parameter_set(self):
        return self.parameter_set

    def get_t_compendium_collections(self):
        return self.t_compendium_collections
        
    def get_m_compendium(self):
        return self.microarray_pipeline.m_compendium
        
    def get_s_compendium(self):
        return self.sequencing_pipeline.s_compendium
        
    def get_t_gene_annotation(self):
        return self.t_gene_annotation
        
    def get_general_parameters(self):
        return self.general_parameters
        
    def get_general_constant(self):
        return self.general_constant
        
    def get_parallel_engine(self):
        return self.parallel_engine
        
        
if __name__ == "__main__":
    #For Testing
    #python transcriptome_pipeline_human_mergedFang.py <target_studies> <target_species> <validate_corr_input> <validate_knowledge_input_samples> <validate_knowledge_input_genes> <(optional)sample_filter_list>
    import pandas as pd
    
    try:
        sample_list_file = sys.argv[1]
        gff_file = sys.argv[2]
        unique_id = sys.argv[3]
    except:
        raise Exception("Usage: python transcriptome_pipeline_20200304.py <sample_list_file> <gene_annotation_file(gff)> <project_name>")


    os.chdir(unique_id)
    tmp = pd.read_csv('../'+sample_list_file)
    exp_list = tmp["Experiment"].tolist()

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
