# Step-by-step example
<h4>This example provides a step-by-step example to make a Salmonella example compendium.</h4>

## Necessary files in this example:
There are some necessary files for compendium building and supervised validation.
<ol>
    <li>Sample list files (<a href = https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/SalmonellaExampleSampleList.csv>SalmonellaExampleSampleList.csv</a>)
        <ul>
            <li>The samples (SRA experiment IDs in SRA database) you are interested in for building a compendium.</li>
        </ul>
    </li>
    <li>Gene annotation files of reference genome (<a href = https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/GCF_000006945.2_ASM694v2> GCF_000006945.2_ASM694v2 </a>)
        <ul>
            <li>A gff file contains metadata information including gene names and positions of all chromosome and plasmid.</li>
            <li>The pipeline will fetch all corresponded sequence from NCBI database using this metadata.</li>
        </ul>
    </li>
    <li>Supervised validation (correlation validation files):
        <ul>
            <li>Samples-studies-conditions mapping table (<a href = https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/Samples_studies_conditions_mappingtable.csv>Samples_studies_conditions_mappingtable.csv</a>)</li>
        </ul>
    </li>
    <li>Supervised validation (knowledge capture validation) files:
        <ol>
            <li>fur gene mutant vs wildtype
                <ul>
                    <li>Selected sample list (<a href = https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/Input_KnowledgeCapture_fur.csv>Input_KnowledgeCapture_fur.csv</a>) </li>
                    <li>Selected gene list (<a href = https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/Input_KnowledgeCapture_fur_related_genes.csv>Input_KnowledgeCapture_fur_related_genes.csv</a>) </li>
                </ul>
            </li>
            <li>hfq gene mutant vs wildtype
                <ul>
                    <li>Selected sample list (<a href = https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/Input_KnowledgeCapture_hfq.csv>Input_KnowledgeCapture_hfq.csv</a>) </li>
                    <li>Selected gene list (<a href = https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/Input_KnowledgeCapture_hfq_related_genes.csv>Input_KnowledgeCapture_hfq_related_genes.csv</a>) </li>
                </ul>
            </li>
        </ol>
    </li>
    <li>Supervised validation (published data comparison) files:
        <ul>
            <li><a href=https://www.sciencedirect.com/science/article/pii/S1931312813004113">Reference compendium for comparison.</a>
                <ul>
                    <li>Original dataset (<a href = https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/SampleTable_GoldenStandard.csv>SampleTable_GoldenStandard.csv</a>) </li>
                    <li>Format refined dataset (<a href = https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/SampleTable_STM_GoldenStandard.csv>SampleTable_STM_GoldenStandard.csv</a>) </li>
                </ul>
            </li>
        </ul>
    </li>
</ol>

## 0. Installation
Please make sure you have installed all the packages, software and set the environment variables correctly. (<a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/tree/Pipeline_20200307>Please refer the main description.</a>)

## 1. Parameters configurations (t_utilities/t_parameters.py)
This step shows you how to modify some important variables which is necessary for building and validating a compendium. To configure the parameters, please open the parameter setting file. (<a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TranscriptomicPipelines/t_utilities/t_parameters.py>t_utilities/t_parameters.py</a>)
<br>There are two classes: 
<ul>
    <li>TranscriptomicConstants: It records the constants, especially some option strings inside the pipeline.</li>
    <li>TranscriptomicParameters: It records the parameters. Some options can be found in the class TranscriptomicConstants.</li>
</ul>

### 1.1 Parallel options
There are four procedures which can be run in parallel:
<ol>
    <li>Reference genome building (corresponded parameter: self.s_value_extraction_refbuild_parameters_parallel_mode)</li>
    <li>Sequencing value extraction (corresponded parameter: self.s_value_extraction_parallel_parameters_parallel_mode)</li>
    <li>Sample mapping (corresponded parameter: self.s_sample_mapping_parallel_parameters_parallel_mode)</li>
    <li>Missing value imputation (corresponded parameter: self.p_imputation_rfimpute_parallel_parameters_parallel_mode)</li>
</ol>

There are two options for parallelization:
<ol>
    <li>local(corresponded option: self.constants.parallel_option_local): The pipeline will run in parallel in the local machine.</li>
    <li>slurm(corresponded option: self.constants.parallel_option_slurm): The pipeline will submit the job to the computation node in cluster via slurm.</li>
</ol>

If you do not have slurm installed in your computer, please change the option as follows in this file:
```
self.s_value_extraction_refbuild_parameters_parallel_mode = self.constants.parallel_option_local
...
self.s_value_extraction_parallel_parameters_parallel_mode = self.constants.parallel_option_local
...
self.s_sample_mapping_parallel_parameters_parallel_mode = self.constants.parallel_option_local
...
self.p_imputation_rfimpute_parallel_parameters_parallel_mode = self.constants.parallel_option_local
```

### 1.2 Number of jobs and number of cores for each node (SLURM mode only) (For Sequencing value extraction and Sample mapping only)
If you can use multiple machine, you can run multiple samples (or runs) at the same time. In addition, you can decide #cores for each job.
<ul>
    <li>Only available in sequencing value extraction and sample mapping because you can split samples or runs into different jobs.</li>
    <li>Number of cores option is only available in sequencing value extraction step because only bowtie2 can use multiple cores for each job.</li>
</ul>


### 1.3 Example configuration (SLURM mode)

#### 1.3.1 Referene genome building
If you want to build the reference genome using 30 cores, you have to adjust the number of cores first. (#job has to be 1 because this step cannot be split into multiple subtasks)

```
self.s_value_extraction_refbuild_parameters_n_jobs_slurm                    = 1 #Cannot be more than one
self.s_value_extraction_refbuild_parameters_slurm_num_core_each_node        = 30
```

In addition, you have to adjust the bowtie2 parameters so that bowtie2 will build the genome with 30 cores:
```
self.s_bowtie2_parameters_build_nthreads                                    = 31
```

#### 1.3.2 Sequencing value extraction
If you want to extract 6 runs, each job use 4 cores at the same time, you may have the following configuration:

```
self.s_value_extraction_parallel_parameters_n_jobs_slurm                    = 6
self.s_value_extraction_parallel_parameters_slurm_num_core_each_node        = 4
```

In addition, you have to adjust the bowtie2 parameters so that bowtie2 will use 4 cores to align the sequencing data for each job:
```
self.s_bowtie2_parameters_align_nthreads                                    = 4
```

#### 1.3.3 Sampling mapping
If you want to process 6 samples at the same time, you may have the following configuration:

```
self.s_sample_mapping_parallel_parameters_n_jobs_slurm                      = 6
self.s_sample_mapping_parallel_parameters_slurm_num_core_each_node          = 30 #Not meaningful in this version
```

The code cannot utilize multiple cores in this step.

#### 1.3.4 Missing value imputation (missing forest only)
Missing value imputation configuration is different than the previous three steps: It does not specify the jobs directly. Instead, it specify number of feature has to be imputed in one job (and number of cores can be used during the imputation).
The lower number of feature has to be imputed in one job, the more jobs will be created. However, the original missing forest imputation is data dependent: the imputation results of the feature is based on the imputation results of the previous imputed feature. Run too many jobs in parallel may negatively impact the imputation performance and may need more iteration to reach convergence.
<br>
If you want to impute 200 features for each job and each job use 32 cores, you may have the following configuration:

```
self.p_imputation_rfimpute_parallel_parameters_n_feature_local              = 200
self.p_imputation_rfimpute_parallel_parameters_n_jobs                       = 1 #Deprecated, not useful
self.p_imputation_rfimpute_parallel_parameters_n_core_local                 = 32
```

### 1.4 Example configuration (LOCAL mode)

#### 1.4.1 Referene genome building
By adjusting the bowtie2 parameters directly, you can speedup the reference genome index building:

```
self.s_bowtie2_parameters_build_nthreads                                    = 31
```

#### 1.3.2 Sequencing value extraction
If you want to extract 2 runs at the same time, each job use 4 cores at the same time, you may have the following configuration:

```
self.s_value_extraction_parallel_parameters_n_processes_local               = 2
```

In addition, you have to adjust the bowtie2 parameters so that bowtie2 will use 4 cores to align the sequencing data for each job:
```
self.s_bowtie2_parameters_align_nthreads                                    = 4
```

Note that it will generate at least 8 (2*4) threads in the local machine. Please make sure your machine have enough computation resources.

#### 1.3.3 Sampling mapping
If you want to process 2 samples at the same time, you may have the following configuration:

```
self.s_sample_mapping_parallel_parameters_n_processes_local                 = 2
```

#### 1.4.4 Missing value imputation (missing forest only)
In local mode, it will just impute one feature at the same time (just as the original missing forest algorithm). However, you can assign more than one cores to speedup each imputation.
<br>
If you want to use 32 cores to speedup a imputation task, you may have the following configuration:

```
self.p_imputation_rfimpute_parallel_parameters_n_core_local                 = 32
```

## 2. Compendium building

### 2.1 Sample list file preparation
You can prepare a sample list file using <a href = https://www.ncbi.nlm.nih.gov/sra>SRA database</a>.
<ol>
    <li>Search the keyword in SRA database.</li>
    <li>Send the searching results into a file using runinfo format.</li>
    <li>Extract the 'Experiment' column (Optional, the output runinfo csv files can also be read by the pipeline).</li>
</ol>

### 2.2 Gene annotation file
You can find a reference genome and download the corresponded gene annotation file using <a href = https://www.ncbi.nlm.nih.gov/assembly>NCBI assembly database</a>.
<ol>
    <li>Search the keyword and select the assembly you are interested in.</li>
    <li>At the right of the page, click 'FTP directory for GenBank assembly'.</li>
    <li>Download the .gff.gz file and then unzip it. That will be the gene annotation file.</li>
</ol>

### 2.3 Run build compendium command
Once you get both sample list file (or sra runinfo file) and gene annotation file, you can run the following command to build the compendium. Suppose we use SalmonellaExample as the project name:

```
python build_compendium_script.py SalmonellaExampleSampleList.csv GCF_000006945.2_ASM694v2 SalmonellaExample
```

It will takes three to four days to download, align, count the gene expressions, merge profiles into one table and normalize it. (Depends on your parallel configuration)
<br>
<a href = https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/SalmonellaExample.tar.gz>The example compendium output can be downloaded here. (Intermediate files are removed)</a>


## 3. Validation

### 3.1 Unsupervised validation
To evaluate a benchmark of this compendium, just run the following command:

```
python unsupervised_validation_script.py SalmonellaExample
```

It will report a benchmark and generate unsupervised validation results (Please refer <a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/VALIDATION.md>validation description</a>).

### 3.2 Supervised validation -- correlation validation

#### 3.2.1 Creating a samples-studies-conditions mapping table
You have to manually curate the metadata to create a sample-studies-conditions mapping table.
<br>
For this <a href = https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/Samples_studies_conditions_mappingtable.csv>example sample-studies-conditions mapping table file</a>, it contains 64 samples from 3 different studies or 40 different conditions.

#### 3.2.2 Run the correlation validation script
Once you prepare the mapping table, run the following command:

```
python supervised_validation_corr_script.py SalmonellaExample Samples_studies_conditions_mappingtable.csv
```

It will generate correlation validation results. (Please refer <a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/VALIDATION.md>validation description</a>).

### 3.3 Supervised validation -- knowledge capture validation

#### 3.3.1 Creating a sample selection file
You have to manually curate the metadata to create a sample selection file.
<br>
For <a href = https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/Input_KnowledgeCapture_fur.csv>fur mutant example</a>, you have to check the metadata and find that two samples (SRX1638996, SRX1638997) are wildtype, and another sample (SRX1638999) is fur mutant.

#### 3.3.2 Creating a gene selection file
You have to survey the published researches to find a gene set which will be regulated by specific stress/mutant and then create a gene selection file
<br>
For <https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/Input_KnowledgeCapture_fur_related_genes.csv>fur regulated gene example</a>, you have to find the <a href=https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5001712/>genes regulated by fur</a>: flagellar genes (activated), SPI1 genes (activated) and SPI2 genes (repressed). In addition, you have to perform necessary gene name conversion to make sure the gene names exist in the compendium.

#### 3.3.3. Run the knowledge capture validation script
Once you prepare both sample selection table and gene selection table, run the following command:

```
python supervised_validation_knowledge_capture_script.py  SalmonellaExample Input_KnowledgeCapture_fur.csv Input_KnowledgeCapture_fur_related_genes.csv
```
It will generate knowledge capture validation results. (Please refer <a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/VALIDATION.md>validation description</a>).

### 3.4 Supervised validation -- published data comparison

#### 3.4.1 Find a reference compendium
You have to find a reference compendium with the samples collected in the compendium for comparison.
<br>
There are three additional steps after you find a reference compendium:
<ol>
    <li>Make sure each row represent different gene and each column represent different samples.</li>
    <li>Gene name mapping: You have to convert the gene names to make sure the gene names is in the compendium.</li>
    <li>Sample mapping: You have to convert the sample tag to sample ID in the compendium.
        <ul>
            <li>Some reference compendia may merge multiple technical replicates into one gene expression profiles. In this case, you can concatenate the technical replicates sample ID with '|' symbol (example: SRX334188|SRX334189). The validation script will merge those technical replicates in the compendium into one gene expression profile.</li>
        </ul>
    </li>
</ol>
<br>
You can compare the format before and after data format conversion:
<ul>
    <li>Original dataset (<a href = https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/SampleTable_GoldenStandard.csv>SampleTable_GoldenStandard.csv</a>) </li>
    <li>Format refined dataset (<a href = https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/SampleTable_STM_GoldenStandard.csv>SampleTable_STM_GoldenStandard.csv</a>) </li>
</ul>

#### 3.4.2 Run the published data comparison script
Once you have well-formatted reference compendium, you can run the following command:

```
python supervised_validation_published_data_comparison_script.py  SalmonellaExample SampleTable_STM_GoldenStandard.csv
```

It will generate published data comparison results. (Please refer <a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/VALIDATION.md>validation description</a>).
