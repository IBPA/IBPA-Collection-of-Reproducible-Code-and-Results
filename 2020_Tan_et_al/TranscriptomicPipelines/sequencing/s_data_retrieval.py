import sys
import Bio.Entrez as Entrez
from enum import Enum

import pandas as pd
import subprocess

import sys
import os

import copy
if (sys.version_info < (3, 0)):
    import s_module_template
    import s_data_retrieval_exceptions
else:
    from . import s_module_template
    from . import s_data_retrieval_exceptions

class SequencingRetrievalConstant(Enum):
    SRADB           = "sra"
    RUNINFO         = "runinfo"
    TEXT            = "text"
    
    NUCLEOTIDEDB    = "nucleotide"
    FASTA           = "fasta"
    
    WRITEMODE       = 'w'
    SPACE           = ' '
    NEWLINE         = '\n'
    
    SRAINFO_COL_RUN         = 'Run'
    SRAINFO_COL_EXP         = 'Experiment'
    SRAINFO_COL_PLATFORM    = "Model"
    SRAINFO_COL_SERIES      = "SRAStudy"
    
    
    

class SequencingRetrievalParameters:
    def __init__(   self,
                    entrez_mail = 'cetan@ucdavis.edu',
                    sra_run_info_path = 'sra_run_info.csv',
                    fasta_path = 'merged.fasta',
                    general_parameters = None):
        self.entrez_mail = entrez_mail
        self.sra_run_info_path = sra_run_info_path
        self.fasta_path = fasta_path
        
        self.skip_srainfo_download = True
        self.skip_fasta_download = True
        
class SequencingRetrievalResults:
    def __init__(self):
        self.fasta_path = None
        self.sra_file_dir = None
        self.mapping_experiment_runs = {}
        self.mapping_experiment_runs_removed = {}
        
    def update_download_metadata(self, fasta_path):
        self.fasta_path = fasta_path

    def update_mapping_experiment_runs(self, mapping_experiment_runs = None):
        self.mapping_experiment_runs = mapping_experiment_runs
        
    def update_mapping_experiment_runs_removed(self, mapping_experiment_runs_removed = None):
        self.mapping_experiment_runs_removed = mapping_experiment_runs_removed
        
    def update_download_data(self, sra_file_dir = ''):
        self.sra_file_dir = sra_file_dir

class SequencingRetrieval(s_module_template.SequencingSubModule):
    def __init__(self, owner):
        self.owner = owner
        self.parameters = SequencingRetrievalParameters(self, general_parameters = self.get_general_parameters())
        self.results = SequencingRetrievalResults()
        
        self.sra_run_info = None
        self.sra_run_info_removed = None
        self.mapping_experiment_runs = {}
        self.mapping_experiment_runs_removed = {}
        
        self.workers = {}
        
        self.configure_parameter_set(general_parameters = self.get_general_parameters())
        
    def do_retrieve(self, platform_id_remove = [], series_id_remove = [], experiment_id_remove = [], run_id_remove = []):
        print("metadata download")
        self.download_metadata()
        print("complete metadata")
        self.complete_data_independent_metadata()
        print("filter")
        self.filter_entry(platform_id_remove, series_id_remove, experiment_id_remove, run_id_remove)
        
        
    def configure_parameter_set(self, general_parameters):
        parameter_set = self.get_parameter_set()
        
        self.parameters.entrez_mail             = parameter_set.s_data_retrieval_parameters_entrez_mail
        self.parameters.sra_run_info_path       = parameter_set.s_data_retrieval_parameters_sra_run_info_path
        self.parameters.fasta_path              = self.get_t_gene_annotation().name + '.fasta'
        
        self.parameters.skip_srainfo_download   = parameter_set.s_data_retrieval_parameters_skip_srainfo_download
        self.parameters.skip_fasta_download     = parameter_set.s_data_retrieval_parameters_skip_fasta_download
        
        
    def get_parameters(self):
        return self.parameters
        
    def get_results(self):
        return self.results
        
    def download_metadata(self):
        #1. Download run_info table from SRA
        #2. Download fasta files from NCBI genome and merge fasta files
        self.download_srainfo()
        self.download_fasta()
        self.results.update_download_metadata(self.get_t_gene_annotation().name + '.fasta')
        
    def download_srainfo(self):
        if self.parameters.skip_srainfo_download == False or self.check_existed_srainfo() == False:
            Entrez.email = self.parameters.entrez_mail
            df_list = []
            id_list = []
            for id in self.get_s_query_id():
                print(Entrez.email)           
                print(id)
                print(SequencingRetrievalConstant.SRADB.value)
                print(SequencingRetrievalConstant.RUNINFO.value)
                print(SequencingRetrievalConstant.TEXT.value)
                id_list.append(id)

                if (len(id_list) > 100):
                    handle = Entrez.efetch( id = id_list, 
                                        db = SequencingRetrievalConstant.SRADB.value, 
                                        rettype = SequencingRetrievalConstant.RUNINFO.value, 
                                        retmode = SequencingRetrievalConstant.TEXT.value)
                
                   
                    df = pd.read_csv(handle)
                    df_list.append(df)
                    handle.close()
                    id_list = []

            if (len(id_list) > 0):
                handle = Entrez.efetch( id = id_list, 
                                        db = SequencingRetrievalConstant.SRADB.value, 
                                        rettype = SequencingRetrievalConstant.RUNINFO.value, 
                                        retmode = SequencingRetrievalConstant.TEXT.value)
                   
                df = pd.read_csv(handle)
                df_list.append(df)
                handle.close()

            self.sra_run_info = pd.concat(df_list)
            self.sra_run_info.to_csv(self.parameters.sra_run_info_path)
        else:
            self.sra_run_info = pd.read_csv(self.parameters.sra_run_info_path)
            

    def check_existed_srainfo(self):
        if not os.path.isfile(self.parameters.sra_run_info_path):
            return False
        else:
            return True
            
    def download_fasta(self):
        if self.parameters.skip_fasta_download == False or self.check_existed_fasta() == False:
            Entrez.email = self.parameters.entrez_mail
            genome_id = self.get_t_gene_annotation().get_genome_id()
            with open(self.parameters.fasta_path, SequencingRetrievalConstant.WRITEMODE.value) as outfile:
                for id in genome_id:
                    print(id)
                    handle = Entrez.efetch( id = id,
                                            db = SequencingRetrievalConstant.NUCLEOTIDEDB.value,
                                            rettype = SequencingRetrievalConstant.FASTA.value,
                                            retmode = SequencingRetrievalConstant.TEXT.value)
                    outfile.write(handle.read())
                    outfile.write(SequencingRetrievalConstant.NEWLINE.value)
        else:
            #Do nothing ==> fasta is ready!
            pass
                                    
    def check_existed_fasta(self):
        if not os.path.isfile(self.get_t_gene_annotation().name + '.fasta'):
            return False
        else:
            return True
        
    def complete_data_independent_metadata(self):
        #Note: You should manage the experiment - run mapping information
        experiments = self.sra_run_info[SequencingRetrievalConstant.SRAINFO_COL_EXP.value].tolist()
        
        metadata = self.get_s_metadata()
        for exp in experiments:
            idx = self.sra_run_info[SequencingRetrievalConstant.SRAINFO_COL_EXP.value] == exp
            self.mapping_experiment_runs[exp] = self.sra_run_info[SequencingRetrievalConstant.SRAINFO_COL_RUN.value][idx].tolist()
        
        self.results.update_mapping_experiment_runs(self.mapping_experiment_runs)
        
        
        #For each row, build the metadata entry
        for index, row in self.sra_run_info.iterrows():
            metadata.new_sequencing_entry(  platform_id = row[SequencingRetrievalConstant.SRAINFO_COL_PLATFORM.value],
                                            series_id = row[SequencingRetrievalConstant.SRAINFO_COL_SERIES.value],
                                            experiment_id = row[SequencingRetrievalConstant.SRAINFO_COL_EXP.value])
        
        df = metadata.get_table()
        df.to_csv("TestMetadataTable.csv")

    def filter_entry(self, platform_id_remove = [], series_id_remove = [], experiment_id_remove = [], run_id_remove = []):
        for exp in self.mapping_experiment_runs:
            for run in self.mapping_experiment_runs[exp]:
                if run in run_id_remove:
                    self.mapping_experiment_runs[exp].remove(run)
            
            if len(self.mapping_experiment_runs[exp]) == 0:
                experiment_id_remove.append(exp)
    
        metadata = self.get_s_metadata()
        for exp in metadata.entries:
            if metadata.entries[exp].platform_id in platform_id_remove:
                experiment_id_remove.append(exp)
            elif metadata.entries[exp].series_id in series_id_remove:
                experiment_id_remove.append(exp)
                
        for exp in experiment_id_remove:
            if exp in metadata.entries.keys():
                metadata.entries[exp].set_remove_ind()
                metadata.entries_removed[exp] = copy.copy(metadata.entries[exp])
                self.mapping_experiment_runs_removed[exp] = copy.copy(self.mapping_experiment_runs[exp])
                del metadata.entries[exp]
                del self.mapping_experiment_runs[exp]
                
        self.results.update_mapping_experiment_runs(self.mapping_experiment_runs)
        self.results.update_mapping_experiment_runs_removed(self.mapping_experiment_runs_removed)
        
        #For Testing
        for exp in metadata.entries:
            print(exp)
            
        for exp in metadata.entries_removed:
            print(exp + '(removed)')
            
        #Update sra information after remove the runs/experiment
        print(self.sra_run_info)
        df_list = []
        for exp in self.mapping_experiment_runs:
            for run in self.mapping_experiment_runs[exp]:
                df_list.append(self.sra_run_info[self.sra_run_info.Run == run])
                
        self.sra_run_info = pd.concat(df_list)
        self.sra_run_info.to_csv(self.parameters.sra_run_info_path)
