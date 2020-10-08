import sys
import copy
if (sys.version_info < (3, 0)):
    import s_module_template
    import s_value_extraction_exceptions
else:
    from . import s_module_template
    from . import s_value_extraction_exceptions

from enum import Enum

import subprocess
import os
import pandas as pd
import pickle
import json
import io

class SequencingExtractionConstant(Enum):
    NEWLINE                     = '\n'
    TABSEP                      = '\t'
    SPACE                       = ' '
    PERCENT                     = '%'
    EXT_SRA                     = '.sra'
    EXT_PAIR_1                  = '_1.fastq.gz'
    EXT_PAIR_2                  = '_2.fastq.gz'
    EXT_SINGLE                  = '.fastq.gz'
    EXT_ALIGN_OUT               = '.sam'
    WRITEMODE                   = 'wb'
    READMODE                    = 'rb'
    SINGLE_PAIREDTYPE           = 'SingleEnd'
    PAIRED_PAIREDTYPE           = 'PairEnd'
    UNKNOWN_PAIREDTYPE          = 'Unknown'
    STRANDEDTYPE                = 'STRANDED'
    UNSTRANDEDTYPE              = 'UNSTRANDED'
    UNKNOWNSTRANDEDTYPE         = 'UNKNOWN'
    STRANDED_MODE               = 'yes'
    UNSTRANDED_MODE             = 'no'
    COUNT_NO_FEATURE            = '__no_feature'
    COUNT_AMBIGUOUS             = '__ambiguous'
    COUNT_TOO_LOW_AQUAL         = '__too_low_aQual'
    COUNT_NOT_ALIGNED           = '__not_aligned'
    COUNT_ALIGNMENT_NOT_UNIQUE  = '__alignment_not_unique'
    SRA_RESULT_OK               = 'consistent'
    INFER_RESULT_UNKNOWN        = 'Unknown'
    JOB_NAME_REFBUILD            = 's_value_extraction_refbuild_'
    JOB_NAME                    = 's_value_extraction_'
    
class SequencingExtractionParallelParameters:
    def __init__(self, default_parallel_parameters):
        self.parallel_parameters = default_parallel_parameters
        self.pyscripts = 'script_get_read_counts_run.py'
    

class SequencingExtractionParameters:
    def __init__(self, working_file_dir = ".",
                    alignment_record_file_ext = ".align_out",
                    infer_experiment_record_file_ext = ".infer_experiment_out",
                    infer_experiment_threshold = 0.9,
                    count_reads_file_ext = '.count_reads',
                    general_parameters = None,
                    n_trial = 10):
        self.working_file_dir = working_file_dir
        self.alignment_record_file_ext = alignment_record_file_ext
        self.infer_experiment_record_file_ext = infer_experiment_record_file_ext
        self.infer_experiment_threshold = infer_experiment_threshold
        self.count_reads_file_ext = count_reads_file_ext
        
        self.n_trial = n_trial
        
        self.skip_all = True
        self.skip_fastq_dump = True
        self.skip_refbuild = True
        self.skip_alignment = True
        self.skip_infer_experiment = True
        self.skip_count_reads = True
        
        self.clean_reference_genome = False
        self.clean_existed_sra_files = True
        self.clean_existed_fastqdump_results = True
        self.clean_existed_alignment_sequence_results = True
        self.clean_existed_alignment_results = True
        self.clean_existed_infer_experiment_results = True
        self.clean_existed_count_read_results = True
        self.clean_existed_worker_file = True
        self.clean_existed_results = False
        
        
        if self.working_file_dir == "":
            raise s_data_retrieval_exceptions.InvalidSRAFilePathException('You should provide the working directory')
        if not self.working_file_dir.endswith(general_parameters.dir_sep):
            self.working_file_dir = self.working_file_dir + general_parameters.dir_sep
        
            
class SequencingExtractionResults:
    def __init__(self):
        self.paired_info = {}
        self.alignment_rate = {}
        self.infer_experiment_result = {}
        self.count_reads_result = {}
        self.count_reads_result_exceptions = {}

        self.sra_file = {} #*.sra
        self.fastq_files = {} #*.fastq.gz
        self.alignment_sequence_file = {} #*.sam
        self.alignment_result_file = {} #*.align_out
        self.infer_experiment_result_file = {} #*.infer_experiment_out
        self.count_reads_file = {} #*.count_reads
        self.worker_file = {} #worker_*
        self.result_file = {} #result_*
        
        self.done = False
        self.exception = None
        
    def update_paired_info(self, run, paired_info):
        self.paired_info[run] = paired_info
        
    def update_alignment_rate(self, run, alignment_rate_run):
        self.alignment_rate[run] = alignment_rate_run

    def update_infer_experiment_result(self, run, infer_experiment_result_run):
        self.infer_experiment_result[run] = infer_experiment_result_run
        
    def update_count_reads_result(self, run, count_reads_result_run, count_reads_result_exceptions_run):
        self.count_reads_result[run] = count_reads_result_run
        self.count_reads_result_exceptions[run] = count_reads_result_exceptions_run
        
    def update_sra_file(self, run, sra_file):
        self.sra_file[run] = sra_file
        
    def update_fastq_files(self, run, fastq_files):
        self.fastq_files[run] = fastq_files
        
    def update_alignment_sequence_file(self, run, alignment_sequence_file):
        self.alignment_sequence_file[run] = alignment_sequence_file
        
    def update_alignment_result_file(self, run, alignment_result_file):
        self.alignment_result_file[run] = alignment_result_file
        
    def update_infer_experiment_result_file(self, run, infer_experiment_result_file):
        self.infer_experiment_result_file[run] = infer_experiment_result_file
        
    def update_count_reads_file(self, run, count_reads_file):
        self.count_reads_file[run] = count_reads_file
        
    def update_worker_file(self, run, worker_file):
        self.worker_file[run] = worker_file
        
    def update_result_file(self, run, result_file):
        self.result_file[run] = result_file
        
    def update_result_from_dict(self, input_dict):
        for attribute in input_dict.keys():
            if attribute == 'infer_experiment_result':
                for run in input_dict['infer_experiment_result'].keys():
                    cur_infer_experiment_result_dict = input_dict['infer_experiment_result'][run]
                    paired_type = cur_infer_experiment_result_dict['paired_type']
                    ratio_failed = cur_infer_experiment_result_dict['ratio_failed']
                    stranded_info = cur_infer_experiment_result_dict['stranded_info']
                    ratio_direction1 = cur_infer_experiment_result_dict['ratio_direction1']
                    ratio_direction2 = cur_infer_experiment_result_dict['ratio_direction2']
                    cur_infer_experiment_result = InferExperimentResults(paired_type, ratio_failed, ratio_direction1, ratio_direction2, stranded_info)
                    self.update_infer_experiment_result(run, cur_infer_experiment_result)
            elif attribute == 'count_reads_result':
                for run in input_dict['count_reads_result'].keys():
                    cur_count_reads_result_dict = input_dict['count_reads_result'][run]
                    cur_count_reads_result_exceptions_dict = input_dict['count_reads_result_exceptions'][run]
                    if (sys.version_info < (3, 0)):
                        cur_count_reads_result = pd.read_csv(io.BytesIO(bytes(cur_count_reads_result_dict)),index_col=0)
                        cur_count_reads_result_exceptions = pd.read_csv(io.BytesIO(bytes(cur_count_reads_result_exceptions_dict)),index_col=0)
                    else:
                        cur_count_reads_result = pd.read_csv(io.BytesIO(bytes(cur_count_reads_result_dict, 'utf-8')),index_col=0)
                        cur_count_reads_result_exceptions = pd.read_csv(io.BytesIO(bytes(cur_count_reads_result_exceptions_dict, 'utf-8')),index_col=0)
                    self.update_count_reads_result(run, cur_count_reads_result, cur_count_reads_result_exceptions)
            elif attribute == 'count_reads_result_exceptions':
                pass
            else:
                if type(input_dict[attribute]) is dict:
                    for run in input_dict[attribute].keys():
                        tmp = getattr(self, attribute)
                        tmp[run] = input_dict[attribute][run]
                else:
                    setattr(self, attribute, input_dict[attribute])
                    
    def save_result_to_dict(self):
        new_json_obj = {}
        for attribute in self.__dict__.keys():
            if attribute == 'infer_experiment_result':
                tmp_result = getattr(self, attribute)
                new_result = {}
                for run in tmp_result:
                    new_result[run] = tmp_result[run].__dict__
            elif attribute == 'count_reads_result' or attribute == 'count_reads_result_exceptions':
                tmp_result = getattr(self, attribute)
                new_result = {}
                for run in tmp_result:
                    new_result[run] = tmp_result[run].to_csv()
            else:
                tmp_result = getattr(self, attribute)
                if type(tmp_result) is dict:
                    new_result = {}
                    for run in tmp_result:
                        new_result[run] = tmp_result[run]
                else:
                    new_result = tmp_result
            
            new_json_obj[attribute] = new_result
            
        return new_json_obj

class InferExperimentResults:
    def __init__(self, paired_type, ratio_failed, ratio_direction1, ratio_direction2, stranded_info):
        self.paired_type = paired_type
        self.ratio_failed = ratio_failed
        self.ratio_direction1 = ratio_direction1
        self.ratio_direction2 = ratio_direction2
        self.stranded_info = stranded_info
        
    def check_stranded_info(self):
        if self.stranded_info == SequencingExtractionConstant.STRANDEDTYPE.value:
            return False
        else:
            return False

class SequencingExtraction(s_module_template.SequencingSubModule):
    def __init__(self, owner):
        self.owner = owner
        self.s_retrieval_results = owner.get_s_data_retrieval_results()
        self.parameters = SequencingExtractionParameters(general_parameters = self.get_general_parameters())
        self.parallel_parameters = SequencingExtractionParallelParameters(self.owner.get_parallel_engine().get_parameters())
        self.results = SequencingExtractionResults()
        self.workers = {}

        self.configure_parameter_set(self.get_general_parameters())
        
    def do_extract(self):
        print("gene_annotation")
        self.configure_parallel_engine_create_bowtie2_index()
        self.prepare_gene_annotation()
        self.reset_parallel_engine()
        
        print("run extraction")
        self.configure_parallel_engine()
        self.prepare_workers()
        self.submit_job()
        self.join_results()
        self.reset_parallel_engine()
        
    def get_parameters(self):
        return self.parameters
        
    def configure_parameter_set(self, general_parameters):
        parameter_set = self.get_parameter_set()
        
        self.parameters.working_file_dir = parameter_set.s_value_extraction_parameters_working_file_dir
        self.parameters.alignment_record_file_ext = parameter_set.s_value_extraction_parameters_alignment_record_file_ext
        self.parameters.infer_experiment_record_file_ext = parameter_set.s_value_extraction_parameters_infer_experiment_record_file_ext
        self.parameters.infer_experiment_threshold = parameter_set.s_value_extraction_parameters_infer_experiment_threshold
        self.parameters.count_reads_file_ext = parameter_set.s_value_extraction_parameters_count_reads_file_ext
        
        self.parameters.n_trial = parameter_set.s_value_extraction_parameters_n_trial
        
        self.parameters.skip_all = parameter_set.s_value_extraction_parameters_skip_all
        self.parameters.skip_fastq_dump = parameter_set.s_value_extraction_parameters_skip_fastq_dump
        self.parameters.skip_refbuild = parameter_set.s_value_extraction_refbuild_parameters_skip_build
        self.parameters.skip_alignment = parameter_set.s_value_extraction_parameters_skip_alignment
        self.parameters.skip_infer_experiment = parameter_set.s_value_extraction_parameters_skip_infer_experiment
        self.parameters.skip_count_reads = parameter_set.s_value_extraction_parameters_skip_count_reads
        
        self.parameters.clean_reference_genome = parameter_set.s_value_extraction_parameters_clean_reference_genome
        self.parameters.clean_existed_sra_files = parameter_set.s_value_extraction_parameters_clean_existed_sra_files
        self.parameters.clean_existed_fastqdump_results = parameter_set.s_value_extraction_parameters_clean_existed_fastqdump_results
        self.parameters.clean_existed_alignment_sequence_results = parameter_set.s_value_extraction_parameters_clean_existed_alignment_sequence_results
        self.parameters.clean_existed_alignment_results = parameter_set.s_value_extraction_parameters_clean_existed_alignment_results
        self.parameters.clean_existed_infer_experiment_results = parameter_set.s_value_extraction_parameters_clean_existed_infer_experiment_results
        self.parameters.clean_existed_count_read_results = parameter_set.s_value_extraction_parameters_clean_existed_count_read_results
        self.parameters.clean_existed_worker_file = parameter_set.s_value_extraction_parameters_clean_existed_worker_file
        self.parameters.clean_existed_results = parameter_set.s_value_extraction_parameters_clean_existed_results
        
        if not self.parameters.working_file_dir.endswith(general_parameters.dir_sep):
            self.parameters.working_file_dir = self.parameters.working_file_dir + general_parameters.dir_sep

        self.parallel_parameters.pyscripts = parameter_set.s_value_extraction_parallel_parameters_pyscript
        
    def get_results(self):
        return self.results
        
    def prepare_gene_annotation(self):
        #Prepare bowtie2 index
        self.create_bowtie2_index()
        self.get_t_gene_annotation().output_bed_file() #For infer strand
        self.get_t_gene_annotation().output_gff_file() #For htseq count
        
    def prepare_workers(self):
        for exp in self.s_retrieval_results.mapping_experiment_runs:
            for run in self.s_retrieval_results.mapping_experiment_runs[exp]:
                self.workers[run] = self.prepare_worker(run)
            
    def prepare_worker(self, run):
        sratool_parameters = self.get_sratool_parameters()
        bowtie2_parameters = self.get_bowtie2_parameters()
        rseqc_parameters = self.get_rseqc_parameters()
        htseq_parameters = self.get_htseq_parameters()
        
        t_gene_annotation = self.get_t_gene_annotation()
        
        general_parameters = self.get_general_parameters()
        general_constant = self.get_general_constant()

        worker = SequencingExtractionWorker(run, 
                                            self.parameters, sratool_parameters, bowtie2_parameters, rseqc_parameters, htseq_parameters,
                                            t_gene_annotation,
                                            general_parameters, general_constant)
        return worker
        
    def reset_parallel_engine(self):
        parallel_engine = self.get_parallel_engine()
        parallel_engine.parameters.reset()
        
    def configure_parallel_engine_create_bowtie2_index(self):
        parameter_set = self.get_parameter_set()
        parallel_engine = self.get_parallel_engine()
        parallel_engine.parameters.set_parallel_mode(parameter_set.s_value_extraction_refbuild_parameters_parallel_mode)
        parallel_engine.parameters.set_n_processes_local(parameter_set.s_value_extraction_refbuild_parameters_n_processes_local)
        parallel_engine.parameters.set_n_jobs_slurm(parameter_set.s_value_extraction_refbuild_parameters_n_jobs_slurm)
        parallel_engine.parameters.set_SLURM_num_node()
        parallel_engine.parameters.set_SLURM_num_core_each_node(parameter_set.s_value_extraction_refbuild_parameters_slurm_num_core_each_node)
        parallel_engine.parameters.set_SLURM_time_limit_hr(parameter_set.s_value_extraction_refbuild_parameters_slurm_time_limit_hr)
        parallel_engine.parameters.set_SLURM_time_limit_min(parameter_set.s_value_extraction_refbuild_parameters_slurm_time_limit_min)
        
    def configure_parallel_engine(self):
        parameter_set = self.get_parameter_set()
        parallel_engine = self.get_parallel_engine()
        parallel_engine.parameters.set_parallel_mode(parameter_set.s_value_extraction_parallel_parameters_parallel_mode)
        parallel_engine.parameters.set_n_processes_local(parameter_set.s_value_extraction_parallel_parameters_n_processes_local)
        parallel_engine.parameters.set_n_jobs_slurm(parameter_set.s_value_extraction_parallel_parameters_n_jobs_slurm)
        parallel_engine.parameters.set_SLURM_num_node()
        parallel_engine.parameters.set_SLURM_num_core_each_node(parameter_set.s_value_extraction_parallel_parameters_slurm_num_core_each_node)
        parallel_engine.parameters.set_SLURM_time_limit_hr(parameter_set.s_value_extraction_parallel_parameters_slurm_time_limit_hr)
        parallel_engine.parameters.set_SLURM_time_limit_min(parameter_set.s_value_extraction_parallel_parameters_slurm_time_limit_min)

    def check_bowtie2_index(self):
        bowtie2_parameters = self.get_bowtie2_parameters()
        file_check = bowtie2_parameters.dir + self.get_t_gene_annotation().get_name() + ".1.bt2"
        if not os.path.isfile(file_check):
            return False
        file_check = bowtie2_parameters.dir + self.get_t_gene_annotation().get_name() + ".2.bt2"
        if not os.path.isfile(file_check):
            return False
        file_check = bowtie2_parameters.dir + self.get_t_gene_annotation().get_name() + ".3.bt2"
        if not os.path.isfile(file_check):
            return False
        file_check = bowtie2_parameters.dir + self.get_t_gene_annotation().get_name() + ".4.bt2"
        if not os.path.isfile(file_check):
            return False
        file_check = bowtie2_parameters.dir + self.get_t_gene_annotation().get_name() + ".rev.1.bt2"
        if not os.path.isfile(file_check):
            return False
        file_check = bowtie2_parameters.dir + self.get_t_gene_annotation().get_name() + ".rev.2.bt2"
        if not os.path.isfile(file_check):
            return False
    
        return True
    
    
    def create_bowtie2_index(self):
        if self.parameters.skip_refbuild == False or self.check_bowtie2_index() == False:
            parallel_engine = self.get_parallel_engine()
            if parallel_engine.parameters.parallel_mode == parallel_engine.parameters.parallel_option.SLURM.value:
                local_command = self.get_bowtie2_build_command()
                command = parallel_engine.get_command_sbatch(SequencingExtractionConstant.JOB_NAME_REFBUILD.value, wait = True)
                print(local_command)
                print(command)
                try:
                    parallel_engine.do_run_slurm_parallel_wait(local_command, command)
                except:
                    raise
            else:
                command = self.get_bowtie2_build_command()
                subprocess.call(command, shell = self.get_general_parameters().use_shell)
            
            
        else:
            print("Skip RefBuild!")
        
    def get_bowtie2_build_command(self):
        bowtie2_parameters = self.get_bowtie2_parameters()
        general_parameters = self.get_general_parameters()
    
        executive_path = bowtie2_parameters.dir + general_parameters.executive_prefix + bowtie2_parameters.build_exe_file + general_parameters.executive_surfix
        thread_par = bowtie2_parameters.build_par_nthreads
        nthread = str(bowtie2_parameters.build_nthreads)
        fasta_path = self.s_retrieval_results.fasta_path
        bowtie2_index_name = self.get_t_gene_annotation().get_name()
        command = [executive_path, thread_par, nthread, fasta_path, bowtie2_index_name]
        
        return(command)
        
    def download_data(self):
        for exp in self.s_retrieval_results.mapping_experiment_runs:
            for run in self.mapping_experiment_runs[exp]:
                self.workers[run].download_data_run_independent()
        
    def prepare_fastq_file(self):
        for exp in self.s_retrieval_results.mapping_experiment_runs:
            for run in self.s_retrieval_results.mapping_experiment_runs[exp]:
                self.workers[run].prepare_fastq_file_run_independent()
                    
    def align_data(self):
        for exp in self.s_retrieval_results.mapping_experiment_runs:
            for run in self.s_retrieval_results.mapping_experiment_runs[exp]:
                self.workers[run].align_data_run_independent()
                
    def infer_stranded_information(self):
        for exp in self.s_retrieval_results.mapping_experiment_runs:
            for run in self.s_retrieval_results.mapping_experiment_runs[exp]:
                self.workers[run].infer_stranded_information_run_independent()
    
    def count_reads(self):
        for exp in self.s_retrieval_results.mapping_experiment_runs:
            for run in self.s_retrieval_results.mapping_experiment_runs[exp]:
                self.workers[run].count_reads_run_independent()
    
    def complete_data_dependent_metadata(self):
        self.data = None #Fake
        
    
    def prepare_worker_file(self, run):
        pickle.dump(self.workers[run], open(self.get_worker_file(run), 'wb'))
        
    def get_worker_file(self, run):
        return 'worker_s_value_extraction_' + str(run) + '.json'
    def get_worker_results_file(self, run):
        return 'results_s_value_extraction_' + str(run) + '.json'
    def get_worker_results(self, run):
        return json.load(open(self.get_worker_results_file(run), 'r'))
        
        
    def get_local_submit_command(self, run):
        if (sys.version_info < (3, 0)):
            python_path = 'python2'
        else:
            python_path = 'python3'
        script_path = self.parallel_parameters.pyscripts
        worker_path = self.get_worker_file(run)
        result_path = self.get_worker_results_file(run)
        
        command = [python_path, script_path, 
                worker_path, 
                result_path]
                
        return command
        
    def submit_job(self):
        if self.parallel_parameters.parallel_parameters.parallel_mode == self.parallel_parameters.parallel_parameters.parallel_option.NONE.value:
            for exp in self.s_retrieval_results.mapping_experiment_runs:
                for run in self.s_retrieval_results.mapping_experiment_runs[exp]:
                    if self.parameters.skip_all == False or self.check_existed_results(run) == False:
                        self.workers[run].do_run()
                        result_file = self.get_worker_results_file(run)
                        json.dump(self.workers[run].results.save_result_to_dict(), open(result_file,'w'))
                    
        elif self.parallel_parameters.parallel_parameters.parallel_mode == self.parallel_parameters.parallel_parameters.parallel_option.LOCAL.value:
            commands = []
            for exp in self.s_retrieval_results.mapping_experiment_runs:
                for run in self.s_retrieval_results.mapping_experiment_runs[exp]:
                    if self.parameters.skip_all == False or self.check_existed_results(run) == False:
                        print(self.parameters.skip_all)
                        print(run)
                        print(self.check_existed_results(run))
                        print(self.get_worker_results(run))
                        print(self.get_worker_results_file(run))
                        raise('?')
                    
                        self.prepare_worker_file(run)
                        commands.append(self.get_local_submit_command(run))
                    
            #Run It !
            parallel_engine = self.get_parallel_engine()
            parallel_engine.do_run_local_parallel(commands)
        elif self.parallel_parameters.parallel_parameters.parallel_mode == self.parallel_parameters.parallel_parameters.parallel_option.SLURM.value:
            local_commands = []
            commands = []
            result_path_list = []
            worker_list = []
            job_name_list = []
            parallel_engine = self.get_parallel_engine()
            for exp in self.s_retrieval_results.mapping_experiment_runs:
                for run in self.s_retrieval_results.mapping_experiment_runs[exp]:
                    if self.parameters.skip_all == False or self.check_existed_results(run) == False:
                        self.prepare_worker_file(run)
                        local_command = self.get_local_submit_command(run)
                        local_commands.append(local_command)
                        job_name = SequencingExtractionConstant.JOB_NAME.value + run
                        command = parallel_engine.get_command_sbatch(job_name)
                        #Run It!
                        commands.append(command)
                        result_path_list.append(self.get_worker_results_file(run))
                        worker_list.append(self.workers[run])
                        job_name_list.append(job_name)
            #Polling
            parallel_engine = self.get_parallel_engine()
            parallel_engine.do_run_slurm_parallel(local_commands, commands, result_path_list, worker_list, job_name_list)
            
    def join_results(self):
        mapping_experiment_runs = self.s_retrieval_results.mapping_experiment_runs
        count_reads_result_2D = {}
        
        exception_occurred = False
        for exp in mapping_experiment_runs:
            count_reads_result_2D[exp] = {}
            for run in mapping_experiment_runs[exp]:
                cur_run_results = SequencingExtractionResults()
                cur_run_results.update_result_from_dict(self.get_worker_results(run))
                if cur_run_results.exception is not None:
                    print(run + ": Exception Occurred : " + str(cur_run_results.exception))
                    exception_occurred = True
                    continue
                
                count_reads_result_2D[exp][run] = cur_run_results.count_reads_result[run]
                #Update the current results
                self.results.update_paired_info(run, cur_run_results.paired_info[run])
                self.results.update_alignment_rate(run, cur_run_results.alignment_rate[run])
                self.results.update_infer_experiment_result(run, cur_run_results.infer_experiment_result[run])
                self.results.update_count_reads_result(run, cur_run_results.count_reads_result[run], cur_run_results.count_reads_result_exceptions[run])
                self.results.update_fastq_files(run, cur_run_results.fastq_files[run])
                self.results.update_alignment_sequence_file(run, cur_run_results.alignment_sequence_file[run])
                self.results.update_alignment_result_file(run, cur_run_results.alignment_result_file[run])
                self.results.update_infer_experiment_result_file(run, cur_run_results.infer_experiment_result_file[run])
                self.results.update_count_reads_file(run, cur_run_results.count_reads_file[run])
                if self.parallel_parameters.parallel_parameters.parallel_mode == self.parallel_parameters.parallel_parameters.parallel_option.NONE.value:
                    self.results.update_worker_file(run, None)
                else:
                    self.results.update_worker_file(run, self.get_worker_file(run))
                self.results.update_result_file(run, self.get_worker_results_file(run))
                
        if exception_occurred == True:
            raise s_value_extraction_exceptions.JoinResultsException('Some exception occurred!')
                
    def check_existed_results(self, run):
        try:
            cur_run_results_dict = self.get_worker_results(run)
        except Exception as e:
            return False
            
        #We have to check the completeness of the results here!
        print(run)
        if cur_run_results_dict['exception'] is not None:
            os.remove(self.get_worker_results_file(run))
            return False
        return True
        
        
class SequencingExtractionWorker:
    def __init__(self, run, 
                parameters, sratool_parameters, bowtie2_parameters, rseqc_parameters, htseq_parameters,
                t_gene_annotation,
                general_parameters, general_constant):
                
        self.run = run
        
        self.parameters = parameters
        self.sratool_parameters = sratool_parameters
        self.bowtie2_parameters = bowtie2_parameters
        self.rseqc_parameters = rseqc_parameters
        self.htseq_parameters = htseq_parameters
        
        self.general_parameters = general_parameters
        self.general_constant = general_constant
        
        self.t_gene_annotation = copy.copy(t_gene_annotation)
        #Clean the actual gene annotation data to save space
        self.t_gene_annotation.gff3_data = None
        self.t_gene_annotation.gff3_data_target_type = None
        
        self.results = SequencingExtractionResults()
        
    def do_run(self):
        cur_n = 0
        while True:
            exception_occurred = False
            self.results.exception = None
            try:
                self.download_data_run_independent()        
            except Exception as e:
                print(str(self.run) + ': Exception Occurred ' + str(e) + ': Try Again')
                self.results.exception = e
                self.clean_intermediate_files_independent(force = True)
                exception_occurred = True
                cur_n = cur_n + 1

            if exception_occurred == False or cur_n == self.parameters.n_trial:
                if cur_n == self.parameters.n_trial:
                    raise Exception('Retrial Failed!')
                break
        
        self.prepare_fastq_file_run_independent()
        self.align_data_run_independent()
        self.infer_stranded_information_run_independent()
        self.count_reads_run_independent()
        self.clean_intermediate_files_independent()


    def get_sratool_prefetch_command(self):
        executive_path = self.sratool_parameters.dir + self.general_parameters.executive_prefix + self.sratool_parameters.prefetch_exe_file + self.general_parameters.executive_surfix
        force_par = self.sratool_parameters.prefetch_par_force
        force = self.sratool_parameters.prefetch_force
        output_par = self.sratool_parameters.prefetch_par_output_file
        output_file_par = "-o"
        output_file_name = self.parameters.working_file_dir + self.run + SequencingExtractionConstant.EXT_SRA.value
        output_dir_par = self.sratool_parameters.prefetch_par_output_dir
        command = [executive_path, self.run, force_par, force, output_file_par, output_file_name]
        return(command)
        
    def get_sratool_vdb_validate_command(self):
        executive_path = self.sratool_parameters.dir + self.general_parameters.executive_prefix + self.sratool_parameters.validate_exe_file + self.general_parameters.executive_surfix
        check_file_name = self.parameters.working_file_dir + self.run + SequencingExtractionConstant.EXT_SRA.value
        command = [executive_path, check_file_name]
        return(command)
        
    def download_data_run_independent(self):
        if self.check_data_independent() == False:
            command = self.get_sratool_prefetch_command()
            print(command)
            subprocess.call(command, stdout=subprocess.PIPE, shell = self.general_parameters.use_shell)
            if self.check_data_independent() == False:
                raise s_value_extraction_exceptions.FailedToDownloadSRAFileException('Failed to download this run:' + self.run)
        
        file_name = self.parameters.working_file_dir + self.run + SequencingExtractionConstant.EXT_SRA.value
        self.results.update_sra_file(self.run, file_name)
        
        
    def check_data_independent(self):
        command = self.get_sratool_vdb_validate_command()
        try:
            binary_result = subprocess.check_output(command, stderr=subprocess.STDOUT, shell = self.general_parameters.use_shell)
        except subprocess.CalledProcessError as e:
            print('FAILED DURING RUNNING VDB_VALIDATE')
            print(e)
            return False
        result = binary_result.decode(self.general_constant.CODEC.value)
        result = result.split(SequencingExtractionConstant.SPACE.value)
        result = result[-1].replace(SequencingExtractionConstant.NEWLINE.value,"")
        if result == SequencingExtractionConstant.SRA_RESULT_OK.value:
            print('GOOD!')
            return True
        else:
            print('FAILED AFTER RUNNING VDB_VALIDATE!')
            print(result)
            print(binary_result)
            return False
        
        
    def prepare_fastq_file_run_independent(self):
        if self.parameters.skip_fastq_dump == False or self.check_existed_fastqdump_results() == False:
            #Do fastq dump if necessary
            command = self.get_fastqdump_command()
            try:
                binary_output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell = self.general_parameters.use_shell)
            except Exception as e:
                print(e)
                raise s_value_extraction_exceptions.FastqdumpFailedException(' '.join(command) + 'Failed to run fastqdump')
            
            try:
                output = binary_output.decode(self.general_constant.CODEC.value)
                output = output.split(SequencingExtractionConstant.NEWLINE.value)
                line_n_read = output[-3]
                line_n_write = output[-2]
                
                n_read = int(line_n_read.split(SequencingExtractionConstant.SPACE.value)[1])
                n_write = int(line_n_write.split(SequencingExtractionConstant.SPACE.value)[1])

                if n_read != n_write:
                    raise s_value_extraction_exceptions.FastqdumpFailedException('n_read != n_write')

            except Exception as e:
                raise s_value_extraction_exceptions.FastqdumpFailedException('Error output from fastqdump')
                
        fastq_files = self.check_paired()
        if len(fastq_files) == 2:
            self.results.update_paired_info(self.run, True)
        elif len(fastq_files) == 1:
            self.results.update_paired_info(self.run, False)
        else:
            raise s_value_extraction_exceptions.FastqdumpFailedException('Output files not found!')
        self.results.update_fastq_files(self.run, fastq_files)
                
    def check_existed_fastqdump_results(self):
        if len(self.check_paired()) > 0:
            return True
        else:
            return False
            
    def check_paired(self):
        file_paired_1 = self.run + SequencingExtractionConstant.EXT_PAIR_1.value
        file_paired_2 = self.run + SequencingExtractionConstant.EXT_PAIR_2.value
        file_single = self.run + SequencingExtractionConstant.EXT_SINGLE.value
    
        if os.path.isfile(file_paired_1) and os.path.isfile(file_paired_2):
            return [file_paired_1, file_paired_2]
        elif os.path.isfile(file_single):
            return [file_single]
        else:
            return []
            

    def align_data_run_independent(self):
        if self.parameters.skip_alignment == False or self.check_existed_alignment_results() == False:
            #Run Bowtie2 if necessary
            command = self.get_bowtie2_align_command()
            try:
                binary_output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell = self.general_parameters.use_shell)
            except Exception as e:
                raise s_value_extraction_exceptions.Bowtie2AlignmentFailedException('Failed to run Bowtie2 alignment')
                
        else:
            #Otherwise just read the results
            with open(self.parameters.working_file_dir + self.run + self.parameters.alignment_record_file_ext, SequencingExtractionConstant.READMODE.value) as infile:
                binary_output = infile.read()
                
        try:
            output = binary_output.decode(self.general_constant.CODEC.value)
            output = output.split(SequencingExtractionConstant.NEWLINE.value)
            
            line_overall_alignment_rate = output[-2]
            alignment_rate = float(line_overall_alignment_rate.split(SequencingExtractionConstant.PERCENT.value)[0])
            if alignment_rate < 0 or alignment_rate > 100:
                raise s_value_extraction_exceptions.Bowtie2AlignmentFailedException('Invalid Bowtie2 alignment results')
        except Exception as e:
            raise s_value_extraction_exceptions.Bowtie2AlignmentFailedException('Invalid Bowtie2 alignment results')
            
        if not self.check_existed_alignment_sequence():
            raise s_value_extraction_exceptions.Bowtie2AlignmentFailedException('SAM output file not exist!')
            
        with open(self.parameters.working_file_dir + self.run + self.parameters.alignment_record_file_ext, SequencingExtractionConstant.WRITEMODE.value) as outfile:
            outfile.write(binary_output)

        self.results.update_alignment_rate(self.run, alignment_rate)
        
        sam_file = self.parameters.working_file_dir + self.run + SequencingExtractionConstant.EXT_ALIGN_OUT.value
        alignment_rate_file = self.parameters.working_file_dir + self.run + self.parameters.alignment_record_file_ext
        self.results.update_alignment_sequence_file(self.run, sam_file)
        self.results.update_alignment_result_file(self.run, alignment_rate_file)
        
        
    def check_existed_alignment_results(self):
        if self.check_existed_alignment_sequence() == False or self.check_existed_alignment_rates() == False:
            return False
        else:
            return True
        
    def check_existed_alignment_sequence(self):
        sam_file = self.parameters.working_file_dir + self.run + SequencingExtractionConstant.EXT_ALIGN_OUT.value
        if not os.path.isfile(sam_file):
            return False
        else:
            return True
            
    def check_existed_alignment_rates(self):
        alignment_rate_file = self.parameters.working_file_dir + self.run + self.parameters.alignment_record_file_ext
        if not os.path.isfile(alignment_rate_file):
            return False
        else:
            return True
            
            
        
    def infer_stranded_information_run_independent(self):
        if self.parameters.skip_infer_experiment == False or self.check_existed_infer_experiment_results() == False:
            #Run infer_experiment.py if necessary
            command = self.get_rseqc_infer_experiment_command()
            try:
                binary_output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell = self.general_parameters.use_shell)
            except Exception as e:
                raise s_value_extraction_exceptions.RSeQCInferExperimentFailedException('Failed to run RSeQC infer_experiment.py')
        else:
            #Otherwise just read the result files
            with open(self.parameters.working_file_dir + self.run + self.parameters.infer_experiment_record_file_ext, SequencingExtractionConstant.READMODE.value) as infile:
                binary_output = infile.read()
        
        try:
            output = binary_output.decode(self.general_constant.CODEC.value)
            output = output.split(SequencingExtractionConstant.NEWLINE.value)
            
            if output[-2].split(SequencingExtractionConstant.SPACE.value)[0] == SequencingExtractionConstant.INFER_RESULT_UNKNOWN.value:
                print('WARNING: Unknown data type: Assume unstranded!')
                paired_type = SequencingExtractionConstant.UNKNOWN_PAIREDTYPE.value
                ratio_failed = -1
                ratio_direction1 = -1
                ratio_direction2 = -1
                stranded_info = SequencingExtractionConstant.UNSTRANDEDTYPE.value
            else:
                paired_type = output[-5].split(SequencingExtractionConstant.SPACE.value)[2]
                if paired_type != SequencingExtractionConstant.SINGLE_PAIREDTYPE.value and paired_type != SequencingExtractionConstant.PAIRED_PAIREDTYPE.value:
                    raise s_value_extraction_exceptions.RSeQCInferExperimentFailedException('Invalid RSeQC infer_experiment results (SINGLE/PAIRED)')
                
                ratio_failed = float(output[-4].split(SequencingExtractionConstant.SPACE.value)[-1])
                if ratio_failed < 0 or ratio_failed > 1:
                    raise s_value_extraction_exceptions.RSeQCInferExperimentFailedException('Invalid RSeQC infer_experiment results (FAILED RATIO)')
                
                ratio_direction1 = float(output[-3].split(SequencingExtractionConstant.SPACE.value)[-1])
                if ratio_direction1 < 0 or ratio_direction1 > 1:
                    raise s_value_extraction_exceptions.RSeQCInferExperimentFailedException('Invalid RSeQC infer_experiment results (DIRECTION 1)')
                
                ratio_direction2 = float(output[-2].split(SequencingExtractionConstant.SPACE.value)[-1])
                if ratio_direction2 < 0 or ratio_direction2 > 1:
                    raise s_value_extraction_exceptions.RSeQCInferExperimentFailedException('Invalid RSeQC infer_experiment results (DIRECTION 2)')
                
                if ratio_direction1 > self.parameters.infer_experiment_threshold or ratio_direction2 > self.parameters.infer_experiment_threshold:
                    stranded_info = SequencingExtractionConstant.STRANDEDTYPE.value
                else:
                    stranded_info = SequencingExtractionConstant.UNSTRANDEDTYPE.value
                
            infer_experiment_result_run = InferExperimentResults(paired_type, ratio_failed, ratio_direction1, ratio_direction2, stranded_info)
            
            
        except Exception as e:
            raise s_value_extraction_exceptions.RSeQCInferExperimentFailedException('Invalid RSeQC infer_experiment results (UNKNOWN)')
            
        
        with open(self.parameters.working_file_dir + self.run + self.parameters.infer_experiment_record_file_ext, SequencingExtractionConstant.WRITEMODE.value) as outfile:
            outfile.write(binary_output)
        
        self.results.update_infer_experiment_result(self.run, infer_experiment_result_run)
        infer_experiment_result_file = self.parameters.working_file_dir + self.run + self.parameters.infer_experiment_record_file_ext
        
        self.results.update_infer_experiment_result_file(self.run, infer_experiment_result_file)

        
    def check_existed_infer_experiment_results(self):
        infer_experiment_result_file = self.parameters.working_file_dir + self.run + self.parameters.infer_experiment_record_file_ext
        if not os.path.isfile(infer_experiment_result_file):
            return False
        else:
            return True
            
    
        
    def count_reads_run_independent(self):
        if self.parameters.skip_count_reads == False or self.check_existed_results() == False:
            #run htseq-count if necessary
            command = self.get_htseq_count_command()
            try:
                binary_output = subprocess.check_output(command, shell = self.general_parameters.use_shell)
            except Exception as e:
                raise s_value_extraction_exceptions.HTSeqCountFailedException('Failed to run htseq-count')
        else:
            with open(self.parameters.working_file_dir + self.run + self.parameters.count_reads_file_ext, SequencingExtractionConstant.READMODE.value) as infile:
                binary_output = infile.read()
                
        try:
            output = binary_output.decode(self.general_constant.CODEC.value)
            count_reads_result_run = pd.DataFrame([x.split(SequencingExtractionConstant.TABSEP.value) for x in output.split(SequencingExtractionConstant.NEWLINE.value)])
            count_reads_result_run = count_reads_result_run.set_index(0)
            count_reads_result_run.rename(columns={1:self.run})
            count_reads_result_run = count_reads_result_run.apply(pd.to_numeric, errors = 'coerce')
            self.check_htseq_count_results(output)
            
                
        except Exception as e:
            raise s_value_extraction_exceptions.HTSeqCountFailedException('Invalid htseq-count results')
        
        with open(self.parameters.working_file_dir + self.run + self.parameters.count_reads_file_ext, SequencingExtractionConstant.WRITEMODE.value) as outfile:
            outfile.write(binary_output)
            
        count_reads_result_exceptions_run = count_reads_result_run.iloc[-6:-1]
        count_reads_result_run = count_reads_result_run.iloc[:-6]
        self.results.update_count_reads_result(self.run,count_reads_result_run, count_reads_result_exceptions_run)
        
        count_reads_file = self.parameters.working_file_dir + self.run + self.parameters.count_reads_file_ext
        
        self.results.update_count_reads_file(self.run, count_reads_file)

    def check_htseq_count_results(self, htseq_count_results):
        output = htseq_count_results.split(SequencingExtractionConstant.NEWLINE.value)
        
        no_feature_line = output[-6]
        if not no_feature_line.startswith(SequencingExtractionConstant.COUNT_NO_FEATURE.value):
            raise s_value_extraction_exceptions.HTSeqCountFailedException('Invalid htseq-count results')
        
        ambiguous_line = output[-5]
        if not ambiguous_line.startswith(SequencingExtractionConstant.COUNT_AMBIGUOUS.value):
            raise s_value_extraction_exceptions.HTSeqCountFailedException('Invalid htseq-count results')
        
        too_low_aQual_line = output[-4]
        if not too_low_aQual_line.startswith(SequencingExtractionConstant.COUNT_TOO_LOW_AQUAL.value):
            raise s_value_extraction_exceptions.HTSeqCountFailedException('Invalid htseq-count results')
            
        not_aligned_line = output[-3]
        if not not_aligned_line.startswith(SequencingExtractionConstant.COUNT_NOT_ALIGNED.value):
            raise s_value_extraction_exceptions.HTSeqCountFailedException('Invalid htseq-count results')
            
        alignment_not_unique_line = output[-2]
        if not alignment_not_unique_line.startswith(SequencingExtractionConstant.COUNT_ALIGNMENT_NOT_UNIQUE.value):
            raise s_value_extraction_exceptions.HTSeqCountFailedException('Invalid htseq-count results')
        
    def check_existed_results(self):
        count_reads_file = self.parameters.working_file_dir + self.run + self.parameters.count_reads_file_ext
        if not os.path.isfile(count_reads_file):
            return False
        else:
            return True
            
    def check_stranded_info(self):
        infer_experiment_result = self.results.infer_experiment_result
        return infer_experiment_result[self.run].check_stranded_info()
        

    def clean_intermediate_files_independent(self, force = False):
        try:
            if self.parameters.clean_existed_sra_files == True or force == True:
                if os.path.isfile(self.results.sra_file[self.run]):
                    os.remove(self.results.sra_file[self.run])
            if self.parameters.clean_existed_fastqdump_results == True or force == True:
                for file in self.results.fastq_files[self.run]:
                    if os.path.isfile(file):
                        os.remove(file)
            if self.parameters.clean_existed_alignment_sequence_results == True or force == True:
                sam_file = self.results.alignment_sequence_file[self.run]
                if os.path.isfile(sam_file):
                    os.remove(sam_file)
            if self.parameters.clean_existed_alignment_results == True or force == True:
                if os.path.isfile(self.results.alignment_result_file[self.run]):
                    os.remove(self.results.alignment_result_file[self.run])
            if self.parameters.clean_existed_infer_experiment_results == True or force == True:
                if os.path.isfile(self.results.infer_experiment_result_file[self.run]):
                    os.remove(self.results.infer_experiment_result_file[self.run])
            if self.parameters.clean_existed_count_read_results == True or force == True:
                if os.path.isfile(self.results.count_reads_file[self.run]):
                    os.remove(self.results.count_reads_file[self.run])
        except:
            pass
        
    
    def get_fastqdump_command(self):
        executive_path = self.sratool_parameters.dir + self.general_parameters.executive_prefix + self.sratool_parameters.fastqdump_exe_file + self.general_parameters.executive_surfix
        fastqdump_par_gzip = self.sratool_parameters.fastqdump_par_gzip
        fastqdump_par_split3 = self.sratool_parameters.fastqdump_par_split3
        fastqdump_par_dir = self.sratool_parameters.fastqdump_par_output_dir
        command = [executive_path, self.parameters.working_file_dir + self.run + SequencingExtractionConstant.EXT_SRA.value, fastqdump_par_gzip, fastqdump_par_split3, fastqdump_par_dir, self.parameters.working_file_dir]
        return(command)
        
        
    def get_bowtie2_align_command(self):
        executive_path = self.bowtie2_parameters.dir + self.general_parameters.executive_prefix + self.bowtie2_parameters.align_exe_file + self.general_parameters.executive_surfix
        
        #Performance Parameters
        thread_par = self.bowtie2_parameters.align_par_nthreads
        nthread = str(self.bowtie2_parameters.align_nthreads)
        reorder_par = self.bowtie2_parameters.align_par_reorder
        mm_par = self.bowtie2_parameters.align_par_mm
        
        align_mode_par = self.bowtie2_parameters.align_mode
        align_speed_par = self.bowtie2_parameters.align_speed
        
        #Bowtie2IndexPath
        index_par = self.bowtie2_parameters.align_par_index_name
        bowtie2_index_name = self.t_gene_annotation.get_name()
        
        #Output
        sam_par = self.bowtie2_parameters.align_par_sam
        output_path = self.parameters.working_file_dir + self.run + SequencingExtractionConstant.EXT_ALIGN_OUT.value
        #Check Paired or Single
        file_paired_1 = self.parameters.working_file_dir + self.run + SequencingExtractionConstant.EXT_PAIR_1.value
        file_paired_2 = self.parameters.working_file_dir + self.run + SequencingExtractionConstant.EXT_PAIR_2.value
        file_single = self.parameters.working_file_dir + self.run + SequencingExtractionConstant.EXT_SINGLE.value
        
        files = self.check_paired()
        
        if len(files) == 2:
            paired_1_par = self.bowtie2_parameters.align_par_paired_1
            paired_2_par = self.bowtie2_parameters.align_par_paired_2
            command = [executive_path, thread_par, nthread, reorder_par, mm_par, 
                        align_mode_par, align_speed_par,
                        index_par, bowtie2_index_name,
                        paired_1_par, files[0], paired_2_par, files[1],
                        sam_par, output_path]
            return(command)
        else:
            unpaired_par = self.bowtie2_parameters.align_par_unpaired
            command = [executive_path, thread_par, nthread, reorder_par, mm_par, 
                        align_mode_par, align_speed_par,
                        index_par, bowtie2_index_name,
                        unpaired_par, files[0],
                        sam_par, output_path]
            return(command)


    def get_rseqc_infer_experiment_command(self):
        executive_path = self.rseqc_parameters.dir + self.general_parameters.executive_prefix + self.rseqc_parameters.infer_experiment_exe_file + self.general_parameters.executive_surfix
        bed_par = self.rseqc_parameters.infer_experiment_par_bed
        bed_path = self.t_gene_annotation.get_bed_file_path()
        input_par = self.rseqc_parameters.infer_experiment_par_input
        input_path = self.parameters.working_file_dir + self.run + SequencingExtractionConstant.EXT_ALIGN_OUT.value
        
        command = [executive_path, bed_par, bed_path, input_par, input_path]
        return(command)
        
    def get_htseq_count_command(self):
        executive_path = self.htseq_parameters.dir + self.general_parameters.executive_prefix + self.htseq_parameters.htseq_count_exe_file + self.general_parameters.executive_surfix
        input_path = self.parameters.working_file_dir + self.run + SequencingExtractionConstant.EXT_ALIGN_OUT.value
        gff_path = self.t_gene_annotation.get_gff_file_path()
        target_type_par = self.htseq_parameters.htseq_count_par_target_type
        target_type = self.t_gene_annotation.get_target_type()
        used_id_par = self.htseq_parameters.htseq_count_par_used_id
        used_id = self.t_gene_annotation.get_used_id()
        quiet_par = self.htseq_parameters.htseq_count_par_quiet
        stranded_par = self.htseq_parameters.htseq_count_par_stranded
        
        if self.check_stranded_info():
            stranded_mode = SequencingExtractionConstant.STRANDED_MODE.value
        else:
            stranded_mode = SequencingExtractionConstant.UNSTRANDED_MODE.value
        
        
        command = [executive_path, input_path, gff_path, target_type_par, target_type, used_id_par, used_id, quiet_par, stranded_par, stranded_mode]
        return(command)
        
    
