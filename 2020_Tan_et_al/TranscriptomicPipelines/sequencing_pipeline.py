import sys
if (sys.version_info < (3, 0)):
    # Python 2 code in this block
    sys.path.insert(0, "sequencing")
    import s_data_retrieval
    import s_value_extraction
    import s_sample_mapping
    import s_gene_mapping
    import s_module_template
else:
    # Python 3
    import sequencing.s_data_retrieval as s_data_retrieval
    import sequencing.s_value_extraction as s_value_extraction
    import sequencing.s_sample_mapping as s_sample_mapping
    import sequencing.s_gene_mapping as s_gene_mapping
    import sequencing.s_module_template as s_module_template

class Bowtie2Parameters:
    def __init__(self, general_constant, general_parameters):
        self.dir = ''
        self.build_exe_file                 = 'bowtie2-build'
        self.build_nthreads                 = 4
        self.build_par_nthreads             = '--threads'
        self.build_index_name               = 'TestTemplate'
        
        self.align_exe_file                 = 'bowtie2'
        self.align_par_index_name           = '-x'
        self.align_par_paired_1             = '-1'
        self.align_par_paired_2             = '-2'
        self.align_par_unpaired             = '-U'
        self.align_par_sam                  = '-S'
        self.align_par_fastq                = '-q'
        self.align_par_phred33              = '--phred33'
        self.align_par_phred64              = '--phred64'
        
        self.align_par_very_fast            = '--very-fast'
        self.align_par_fast                 = '--fast'
        self.align_par_sensitive            = '--sensitive'
        self.align_par_very_sensitive       = '--very-sensitive'
        self.align_par_very_fast_local      = '--very-fast-local'
        self.align_par_fast_local           = '--fast-local'
        self.align_par_sensitive_local      = '--sensitive-local'
        self.align_par_very_sensitive_local = '--very-sensitive-local'
        self.align_speed                    = self.align_par_sensitive
        
        self.align_par_end_to_end           = '--end-to-end'
        self.align_par_local                = '--local'
        self.align_mode                     = self.align_par_end_to_end
        
        self.align_par_nthreads             = '--threads'
        self.align_nthreads                 = 4
        self.align_par_reorder              = '--reorder'
        self.align_par_mm                   = '--mm'
        
        self.dir.replace(general_constant.UNIX_DIR_SEP.value,general_parameters.dir_sep)
        self.dir.replace(general_constant.WINDOWS_DIR_SEP.value,general_parameters.dir_sep)
        
        if not self.dir.endswith(general_parameters.dir_sep) and self.dir != "":
            self.dir = self.dir + general_parameters.dir_sep
            
        
        
        
        
class SRAToolkitParameters:
    def __init__(self, general_constant, general_parameters):
        self.dir = ''
        self.prefetch_exe_file          = 'prefetch'
        self.prefetch_par_output_file   = '-o'
        self.prefetch_par_force         = '-f'
        self.prefetch_force             = 'all'
        
        self.validate_exe_file          = 'vdb-validate'
        
        self.fastqdump_exe_file         = 'fastq-dump'
        self.fastqdump_par_gzip         = '--gzip'
        self.fastqdump_par_split3       = '--split-3'
        self.fastqdump_par_output_dir  = '-O'

        self.dir.replace(general_constant.UNIX_DIR_SEP.value,general_parameters.dir_sep)
        self.dir.replace(general_constant.WINDOWS_DIR_SEP.value,general_parameters.dir_sep)
        
        if not self.dir.endswith(general_parameters.dir_sep) and self.dir != "":
            self.dir = self.dir + general_parameters.dir_sep
            

class RSeQCParameters:
    def __init__(self, general_constant, general_parameters):
        self.dir = ''
        self.infer_experiment_exe_file = 'infer_experiment.py'
        self.infer_experiment_par_bed = '-r'
        self.infer_experiment_par_input = '-i'
        
        self.dir.replace(general_constant.UNIX_DIR_SEP.value,general_parameters.dir_sep)
        self.dir.replace(general_constant.WINDOWS_DIR_SEP.value,general_parameters.dir_sep)
        
        if not self.dir.endswith(general_parameters.dir_sep) and self.dir != "":
            self.dir = self.dir + general_parameters.dir_sep
            
class HTSeqParameters:
    def __init__(self, general_constant, general_parameters):
        self.dir = ''
        self.htseq_count_exe_file = 'htseq-count'
        self.htseq_count_par_target_type = '-t'
        self.htseq_count_par_used_id = '-i'
        self.htseq_count_par_quiet = '-q'
        self.htseq_count_par_stranded = '-s'
        
        self.dir.replace(general_constant.UNIX_DIR_SEP.value,general_parameters.dir_sep)
        self.dir.replace(general_constant.WINDOWS_DIR_SEP.value,general_parameters.dir_sep)
        
        if not self.dir.endswith(general_parameters.dir_sep) and self.dir != "":
            self.dir = self.dir + general_parameters.dir_sep
            
        
class SequencingPipeline(s_module_template.SequencingModule):
    def __init__(self, owner, s_query_id, s_compendium):
        self.owner = owner
        
        self.s_query_id = s_query_id
        self.s_compendium = s_compendium
        self.s_compendium.metadata.configure_sequencing()
        
        self.bowtie2_parameters = Bowtie2Parameters(self.get_general_constant(), self.get_general_parameters())
        self.sratool_parameters = SRAToolkitParameters(self.get_general_constant(), self.get_general_parameters())
        self.rseqc_parameters = RSeQCParameters(self.get_general_constant(), self.get_general_parameters())
        self.htseq_parameters = HTSeqParameters(self.get_general_constant(), self.get_general_parameters())
        
        self.s_data_retrieval = s_data_retrieval.SequencingRetrieval(self)
        self.s_value_extraction = s_value_extraction.SequencingExtraction(self)
        self.s_sample_mapping = s_sample_mapping.SequencingSampleMapping(self)
        self.s_gene_mapping = s_gene_mapping.SequencingGeneMapping(self)
        
        self.configure_parameter_set(self.get_general_constant(), self.get_general_parameters())
        
    def configure_parameter_set_all(self, general_constant, general_parameters):
        self.configure_parameter_set(general_constant, general_parameters)
        self.s_data_retrieval.configure_parameter_set(general_parameters)
        self.s_value_extraction.configure_parameter_set(general_parameters)
        self.s_sample_mapping.configure_parameter_set()
        self.s_gene_mapping.configure_parameter_set()
        
    def configure_parameter_set(self, general_constant, general_parameters):
        parameter_set = self.get_parameter_set()
        self.configure_sratool_parameter_set(parameter_set, general_constant, general_parameters)
        self.configure_bowtie2_parameter_set(parameter_set, general_constant, general_parameters)
        self.configure_rseqc_parameter_set(parameter_set, general_constant, general_parameters)
        self.configure_htseq_parameter_set(parameter_set, general_constant, general_parameters)
        
    def configure_sratool_parameter_set(self, parameter_set, general_constant, general_parameters):
        self.sratool_parameters.dir                        = parameter_set.s_sratool_parameters_dir
        self.sratool_parameters.prefetch_exe_file          = parameter_set.s_sratool_parameters_prefetch_exe_file
        self.sratool_parameters.prefetch_par_output_file   = parameter_set.constants.s_sratool_parameters_prefetch_par_output_file
        self.sratool_parameters.prefetch_par_force         = parameter_set.constants.s_sratool_parameters_prefetch_par_force
        self.sratool_parameters.prefetch_force             = parameter_set.s_sratool_parameters_prefetch_force
        self.sratool_parameters.prefetch_par_output_dir    = parameter_set.constants.s_sratool_parameters_prefetch_par_output_dir
        
        self.sratool_parameters.validate_exe_file          = parameter_set.s_sratool_parameters_validate_exe_file
        
        self.sratool_parameters.fastqdump_exe_file         = parameter_set.s_sratool_parameters_fastqdump_exe_file
        self.sratool_parameters.fastqdump_par_gzip         = parameter_set.constants.s_sratool_parameters_fastqdump_par_gzip
        self.sratool_parameters.fastqdump_par_split3       = parameter_set.constants.s_sratool_parameters_fastqdump_par_split3
        self.sratool_parameters.fastqdump_par_output_dir   = parameter_set.constants.s_sratool_parameters_fastqdump_par_output_dir

        self.sratool_parameters.dir.replace(general_constant.UNIX_DIR_SEP.value,general_parameters.dir_sep)
        self.sratool_parameters.dir.replace(general_constant.WINDOWS_DIR_SEP.value,general_parameters.dir_sep)
        
        if not self.sratool_parameters.dir.endswith(general_parameters.dir_sep) and self.sratool_parameters.dir != "":
            self.sratool_parameters.dir = self.sratool_parameters.dir + general_parameters.dir_sep
        
            
    def configure_bowtie2_parameter_set(self, parameter_set, general_constant, general_parameters):
        self.bowtie2_parameters.dir                            = parameter_set.s_bowtie2_parameters_dir
        self.bowtie2_parameters.build_exe_file                 = parameter_set.s_bowtie2_parameters_build_exe_file
        self.bowtie2_parameters.build_nthreads                 = parameter_set.s_bowtie2_parameters_build_nthreads
        self.bowtie2_parameters.build_par_nthreads             = parameter_set.constants.s_bowtie2_parameters_build_par_nthreads
        #self.bowtie2_parameters.build_index_name               = parameter_set.s_bowtie2_parameters_build_index_name
        
        self.bowtie2_parameters.align_exe_file                 = parameter_set.s_bowtie2_parameters_align_exe_file
        self.bowtie2_parameters.align_par_index_name           = parameter_set.constants.s_bowtie2_parameters_align_par_index_name
        self.bowtie2_parameters.align_par_paired_1             = parameter_set.constants.s_bowtie2_parameters_align_par_paired_1
        self.bowtie2_parameters.align_par_paired_2             = parameter_set.constants.s_bowtie2_parameters_align_par_paired_2
        self.bowtie2_parameters.align_par_unpaired             = parameter_set.constants.s_bowtie2_parameters_align_par_unpaired
        self.bowtie2_parameters.align_par_sam                  = parameter_set.constants.s_bowtie2_parameters_align_par_sam
        self.bowtie2_parameters.align_par_fastq                = parameter_set.constants.s_bowtie2_parameters_align_par_fastq
        self.bowtie2_parameters.align_par_phred33              = parameter_set.constants.s_bowtie2_parameters_align_par_phred33
        self.bowtie2_parameters.align_par_phred64              = parameter_set.constants.s_bowtie2_parameters_align_par_phred64
        
        self.bowtie2_parameters.align_par_very_fast            = parameter_set.constants.s_bowtie2_parameters_align_par_very_fast
        self.bowtie2_parameters.align_par_fast                 = parameter_set.constants.s_bowtie2_parameters_align_par_fast
        self.bowtie2_parameters.align_par_sensitive            = parameter_set.constants.s_bowtie2_parameters_align_par_sensitive
        self.bowtie2_parameters.align_par_very_sensitive       = parameter_set.constants.s_bowtie2_parameters_align_par_very_sensitive
        self.bowtie2_parameters.align_par_very_fast_local      = parameter_set.constants.s_bowtie2_parameters_align_par_very_fast_local
        self.bowtie2_parameters.align_par_fast_local           = parameter_set.constants.s_bowtie2_parameters_align_par_fast_local
        self.bowtie2_parameters.align_par_sensitive_local      = parameter_set.constants.s_bowtie2_parameters_align_par_sensitive_local
        self.bowtie2_parameters.align_par_very_sensitive_local = parameter_set.constants.s_bowtie2_parameters_align_par_very_sensitive_local
        self.bowtie2_parameters.align_speed                    = parameter_set.s_bowtie2_parameters_align_speed
        
        self.bowtie2_parameters.align_par_end_to_end           = parameter_set.constants.s_bowtie2_parameters_align_par_end_to_end
        self.bowtie2_parameters.align_par_local                = parameter_set.constants.s_bowtie2_parameters_align_par_local
        self.bowtie2_parameters.align_mode                     = parameter_set.s_bowtie2_parameters_align_mode
        
        self.bowtie2_parameters.align_par_nthreads             = parameter_set.constants.s_bowtie2_parameters_align_par_nthreads
        self.bowtie2_parameters.align_nthreads                 = parameter_set.s_bowtie2_parameters_align_nthreads
        self.bowtie2_parameters.align_par_reorder              = parameter_set.constants.s_bowtie2_parameters_align_par_reorder
        self.bowtie2_parameters.align_par_mm                   = parameter_set.constants.s_bowtie2_parameters_align_par_mm
        
        self.bowtie2_parameters.dir.replace(general_constant.UNIX_DIR_SEP.value,general_parameters.dir_sep)
        self.bowtie2_parameters.dir.replace(general_constant.WINDOWS_DIR_SEP.value,general_parameters.dir_sep)
        
        if not self.bowtie2_parameters.dir.endswith(general_parameters.dir_sep) and self.bowtie2_parameters.dir != "":
            self.bowtie2_parameters.dir = self.bowtie2_parameters.dir + general_parameters.dir_sep
    
    def configure_rseqc_parameter_set(self, parameter_set, general_constant, general_parameters):
        self.rseqc_parameters.dir = parameter_set.s_rseqc_parameters_dir
        self.rseqc_parameters.infer_experiment_exe_file = parameter_set.s_rseqc_parameters_infer_experiment_exe_file
        self.rseqc_parameters.infer_experiment_par_bed = parameter_set.constants.s_rseqc_parameters_infer_experiment_par_bed
        self.rseqc_parameters.infer_experiment_par_input = parameter_set.constants.s_rseqc_parameters_infer_experiment_par_input
        
        self.rseqc_parameters.dir.replace(general_constant.UNIX_DIR_SEP.value,general_parameters.dir_sep)
        self.rseqc_parameters.dir.replace(general_constant.WINDOWS_DIR_SEP.value,general_parameters.dir_sep)
        
        if not self.rseqc_parameters.dir.endswith(general_parameters.dir_sep) and self.rseqc_parameters.dir != "":
            self.rseqc_parameters.dir = self.rseqc_parameters.dir + general_parameters.dir_sep
            
    
    def configure_htseq_parameter_set(self, parameter_set, general_constant, general_parameters):
        self.htseq_parameters.dir = parameter_set.s_htseq_parameters_dir
        self.htseq_parameters.htseq_count_exe_file = parameter_set.s_htseq_parameters_htseq_count_exe_file
        self.htseq_parameters.htseq_count_par_target_type = parameter_set.constants.s_htseq_parameters_htseq_count_par_target_type
        self.htseq_parameters.htseq_count_par_used_id = parameter_set.constants.s_htseq_parameters_htseq_count_par_used_id
        self.htseq_parameters.htseq_count_par_quiet = parameter_set.constants.s_htseq_parameters_htseq_count_par_quiet
        self.htseq_parameters.htseq_count_par_stranded = parameter_set.constants.s_htseq_parameters_htseq_count_par_stranded
        
        self.htseq_parameters.dir.replace(general_constant.UNIX_DIR_SEP.value,general_parameters.dir_sep)
        self.htseq_parameters.dir.replace(general_constant.WINDOWS_DIR_SEP.value,general_parameters.dir_sep)
        
        if not self.htseq_parameters.dir.endswith(general_parameters.dir_sep) and self.htseq_parameters.dir != "":
            self.htseq_parameters.dir = self.htseq_parameters.dir + general_parameters.dir_sep
        
        
    def run_sequencing_pipeline(self, platform_id_remove = [], series_id_remove = [], experiment_id_remove = [], run_id_remove = []):
        self.s_data_retrieval.download_metadata()
        self.s_data_retrieval.complete_data_independent_metadata()
        self.s_data_retrieval.filter_entry(platform_id_remove, series_id_remove, experiment_id_remove, run_id_remove)
        
        #Parallel (run level)
        self.s_value_extraction.prepare_gene_annotation()
        self.s_value_extraction.prepare_workers()
        self.s_value_extraction.submit_job()
        self.s_value_extraction.join_results()
        
        #Parallel (exp level)
        self.s_sample_mapping.prepare_workers()
        self.s_sample_mapping.submit_job()
        self.s_sample_mapping.join_results()
        #Now the thing are the same as serial now :)
        self.s_sample_mapping.merge_sample()
        self.s_sample_mapping.complete_data_dependent_metadata()
        self.s_gene_mapping.map_gene()
        
    def get_s_query_id(self):
        return self.s_query_id
        
    def get_s_metadata(self):
        return self.s_compendium.get_metadata()
        
    def get_s_data(self):
        return self.s_compendium.get_data()
        
    def get_s_data_retrieval_results(self):
        return self.s_data_retrieval.get_results()
        
    def get_s_data_retrieval_parameters(self):
        return self.s_data_retrieval.get_parameters()
        
    def get_s_value_extraction_results(self):
        return self.s_value_extraction.get_results()
        
    def get_s_sample_mapping_results(self):
        return self.s_sample_mapping.get_results()
        
    def get_bowtie2_parameters(self):
        return self.bowtie2_parameters
    
    def get_sratool_parameters(self):
        return self.sratool_parameters
        
    def get_rseqc_parameters(self):
        return self.rseqc_parameters
        
    def get_htseq_parameters(self):
        return self.htseq_parameters
        
