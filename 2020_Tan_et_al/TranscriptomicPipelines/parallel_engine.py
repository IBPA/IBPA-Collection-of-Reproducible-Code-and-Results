import json

import time
import copy
import subprocess

from enum import Enum

import sequencing_pipeline

class ParallelOptions(Enum):
    NONE = "None"
    LOCAL = "local"
    SLURM = "SLURM"
    
class ParallelParameters_SLURM:
    def __init__(self):
        self.par_num_node = '-N'
        self.num_node = 1
        self.par_num_core_each_node = '-c'
        self.num_core_each_node = 1
        self.par_time_limit = '--time'
        self.time_limit_hr = 10
        self.time_limit_min = 0
        self.par_job_name = '-J'

        self.par_output = '-o'
        self.output_ext = ".output"
        self.par_error = '-e'
        self.error_ext = ".error"
        
        self.par_wait = '-W'
        
        self.shell_script_path = 'job.sh'
        
        self.queue_par_job_name = '-n'
        self.queue_par_noheader = '-h'

class ParallelParameters:
    def __init__(self):
        self.parallel_option = ParallelOptions
        self.parallel_mode = self.parallel_option.SLURM.value
        self.n_processes_local = 2
        self.n_jobs_slurm = 8
        self.parameters_SLURM = ParallelParameters_SLURM()
        
    def set_parallel_mode(self, mode = ParallelOptions.SLURM.value):
        self.parallel_mode = mode
        
    def set_n_processes_local(self, n_processes_local = 2):
        self.n_processes_local = n_processes_local
    
    def set_n_jobs_slurm(self, n_jobs_slurm = 8):
        self.n_jobs_slurm = n_jobs_slurm
        
    def set_SLURM_num_node(self, num_node = 1):
        self.parameters_SLURM.num_node = num_node
    
    def set_SLURM_num_core_each_node(self, num_core_each_node = 1):
        self.parameters_SLURM.num_core_each_node = num_core_each_node
        
    def set_SLURM_time_limit_hr(self, time_limit_hr = 10):
        self.parameters_SLURM.time_limit_hr = time_limit_hr
        
    def set_SLURM_time_limit_min(self, time_limit_min = 0):
        self.parameters_SLURM.time_limit_min = time_limit_min
        
    def reset(self):
        self.parallel_option = ParallelOptions
        self.parallel_mode = self.parallel_option.SLURM.value
        self.n_processes_local = 2
        self.n_jobs_slurm = 8
        self.parameters_SLURM = ParallelParameters_SLURM()


        
class ParallelEngine:
    def __init__(self):
        self.parameters = ParallelParameters()
        
    def reset_parameters(self):
        self.parameters = ParallelParameters()
        
    def get_parameters(self):
        return copy.copy(self.parameters)
        
    def set_parameters(self, parameters):
        self.parameters = parameters
        
    def prepare_shell_file(self, local_submit_command):
        command = ['srun']
        command.extend(local_submit_command)
        command_shell =' '.join(command)
        
        shell_script_path = self.parameters.parameters_SLURM.shell_script_path
        
        with open(shell_script_path, 'w') as tmp:
            tmp.writelines('#!/bin/bash\n')
            tmp.writelines(command_shell)
            
    def get_command_srun(self):
        exe_path = 'srun'
        slurm_parameters = self.parameters.parameters_SLURM
        par_num_node = slurm_parameters.par_num_node
        num_node = str(slurm_parameters.num_node)
        par_num_core_each_node = slurm_parameters.par_num_core_each_node
        num_core_each_node = str(slurm_parameters.num_core_each_node)
        par_time_limit = slurm_parameters.par_time_limit
        time_limit = str(slurm_parameters.time_limit_hr) + ":" + (format(slurm_parameters.time_limit_min,'02')) + ":00"
        return ([exe_path, par_num_node, num_node, par_num_core_each_node, num_core_each_node, par_time_limit, time_limit])
        
    def get_command_sbatch(self, job_name, wait = False):
        exe_path = 'sbatch'
        
        slurm_parameters = self.parameters.parameters_SLURM
        
        par_num_node = slurm_parameters.par_num_node
        num_node = str(slurm_parameters.num_node)
        par_num_core_each_node = slurm_parameters.par_num_core_each_node
        num_core_each_node = str(slurm_parameters.num_core_each_node)
        par_time_limit = slurm_parameters.par_time_limit
        time_limit = str(slurm_parameters.time_limit_hr) + ":" + (format(slurm_parameters.time_limit_min,'02')) + ":00"
        par_job_name = slurm_parameters.par_job_name
        par_output = slurm_parameters.par_output
        output_file = job_name + slurm_parameters.output_ext
        par_error = slurm_parameters.par_error
        error_file = job_name + slurm_parameters.error_ext
        
        par_wait = slurm_parameters.par_wait
        
        shell_script_path = slurm_parameters.shell_script_path
        if wait == False:
            return ([exe_path, par_num_node, num_node, par_num_core_each_node, num_core_each_node, par_time_limit, time_limit, \
                    par_job_name, job_name, par_output, output_file, par_error, error_file, shell_script_path])
        else:
            return ([exe_path, par_num_node, num_node, par_num_core_each_node, num_core_each_node, par_time_limit, time_limit, \
                    par_job_name, job_name, par_wait, par_output, output_file, par_error, error_file, shell_script_path])
    
    def do_run_slurm_parallel_wait(self, local_command, command):
        self.prepare_shell_file(local_command)
        try:
            subprocess.check_output(command)
        except:
            raise
    
    def do_run_local_parallel(self, command_list):
        obj_Popen = [None]*self.parameters.n_processes_local
        next_entry_idx = 0
        while True:
            finish = True
            for i in range(self.parameters.n_processes_local):
                if obj_Popen[i] is not None:
                    if obj_Popen[i].poll() is None:
                        #Working
                        finish = False
                        continue
                    else:
                        #Join the zombie process
                        obj_Popen[i].wait()
                        obj_Popen[i] = None
                        
                if next_entry_idx < len(command_list):
                    #New job has to be assigned
                    obj_Popen[i] = subprocess.Popen(command_list[next_entry_idx])
                    next_entry_idx = next_entry_idx + 1
                    finish = False
                    
            if finish == True:
                break
                
                
                
    def do_run_slurm_parallel(self, local_command_list, command_list, result_path_list, worker_list, job_name_list):
        slurm_parameters = self.parameters.parameters_SLURM

        running_idx = [-1]*self.parameters.n_jobs_slurm
        next_entry_idx = 0
        
        while True:
            finish = True
            time.sleep(1)
            print(running_idx)
            for i in range(self.parameters.n_jobs_slurm):
                if running_idx[i] != -1:
                    cur_job_name = job_name_list[running_idx[i]]
                    query_result = subprocess.check_output(['squeue',slurm_parameters.queue_par_job_name,cur_job_name,slurm_parameters.queue_par_noheader])
                    if len(query_result) > 0:
                        #Working ==> Wait
                        finish = False
                        continue
                    else:
                        #Stop Working: Check
                        try:
                            cur_results = json.load(open(result_path_list[running_idx[i]],'r'))
                            if cur_results['done'] == False:
                                #Failed: Incomplete
                                print("FAILED (Incomplete Result):" + cur_job_name)
                            else:
                                #Complete
                                print("FINISHED! : " + cur_job_name)
                                
                                
                        except Exception as e:
                            #Failed: No File
                            print("FAILED (No Result):" + cur_job_name)
                            print(query_result)
                            #raise Exception
                        running_idx[i] = -1
                        #worker_list[running_idx[i]].clean_intermediate_files_independent(force = True) #Comment it for testing
                        
                        
                if next_entry_idx < len(command_list):
                    #New job has to be assigned
                    running_idx[i] = next_entry_idx
                    self.prepare_shell_file(local_command_list[next_entry_idx])
                    subprocess.call(command_list[next_entry_idx])
                    cur_job_name = job_name_list[running_idx[i]]
                    retry_idx = 0
                    retry_num = 10
                    print(command_list[next_entry_idx])
                    while True:
                        query_result = subprocess.check_output(['squeue',slurm_parameters.queue_par_job_name,cur_job_name,slurm_parameters.queue_par_noheader])
                        print(query_result)
                        if len(query_result) > 0:
                            break
                        time.sleep(1)
                        retry_idx = retry_idx + 1
                        if retry_idx > retry_num:
                            print("FAILED (Failed to start):" + cur_job_name)
                    
                    next_entry_idx = next_entry_idx + 1
                    print('New')
                    finish = False
                    
            if finish == True:
                break
                
                
