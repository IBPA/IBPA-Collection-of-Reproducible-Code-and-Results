# Automated Omics Compendium Preparation Pipeline

## Propose
This toolkit can prepare the transcriptomic compendium (a normalized, format-consistent data matrix across samples from different studies) by collecting the samples in <a href="https://www.ncbi.nlm.nih.gov/sra">Sequencing Read Archive (SRA) </a> database given the topic you are interested in and your target species.

(In the future, this toolkit will be capable to process microarray dataset from GEO and ArrayExpress database)

## Steps of preparing a transcriptomic compendium
The pipeline will do the necessary work for building transcriptomic compendium for you in five steps:<br>
<b>(To check the exact format, please read <a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/STEP-BY-STEP.md>Step-by-Step example</a>)</b>

### 1. Metadata preparation
<h4>This step will take two user inputs to prepare all necessary metadata for sequencing data processing: </h4>
    <ol>
        <li>Sample List: The list that contains samples (experiment ID in SRA database) you are interested in.</li>
        <li>Gene Annotation File: A GFF file downloaded from NCBI genome database. This annotation file allow the pipeline to fetch reference genome sequence and extract the corresponded gene names.</li>
    </ol>
<h4>The output metadata will contain all necessary information for sequencing data processing: </h4>
    <ol>
        <li>Run information in SRA database: It contains corresponded run information for samples you are interested in. One sample (experiment ID) may contain more than one runs.</li>
        <li>Reference genome files: Files in Bowtie2 index format to allow the pipeline align the sequencing data with this reference.</li>
        <li>Reference genome sequence direction information: A BED file to allow the pipeline detect the sequencing data type (stranded or unstranded). </li>
    </ol>
    
### 2. Sequencing data download
<h4>This step will take run information input and then download all sequencing data of samples you are interested in from SRA database: </h4>
    <ul>
        <li>Run information in SRA database: Generated from step (1) and contains corresponded run information for samples you are interested in.</li>
    </ul>
<h4> The output files are downloaded and format-converted sequencing data: </h4>
    <ul>
        <li>Sequencing data: Fastq files for each run. Two files for one run if this run is paired-end data, otherwise each run will generate one fastq file.
    </ul>

### 3. Sequencing data alignment
<h4>This step will take sequencing data and reference genome files as inputs to perform sequence alignment:</h4>
    <ol>
        <li>Sequencing data: Generated from step (2). Fastq files for each run.</li>
        <li>Reference genome files: Generated from step(1). Reference genome sequence for alignment.</li>
    </ol>
<h4>The output file is alignment results is SAM format and alignment rates:</h4>
    <ol>
        <li>The alignment result files: For each run, a file contains the alignment result in SAM format is generated.</li>
        <li>Alignment rate information: Alignment rate information will be recorded (for internal use only).</li>
    </ol>

### 4. Gene expression counting
<h4>This step will take gene alignment results, sequence direction information file (BED file) and gene annotation file (GFF file) as inputs to generate the gene expression profile for each run.<br></h4>
    <ol>
        <li>The alignment result files: Files in SAM format generated from step (3) which recorded the alignment result.</li>
        <li>BED file contains sequence direction information: A file in BED format generated from step (1). With this sequence direction information, the pipeline can detect whether sequencing data is stranded or unstranded.</li>
        <li>GFF file contains gene annotation information: A file in GFF format given from users. With this information, the pipeline can generate gene expression profiles with correct gene names.</li>
    </ol>
<h4>The output file are the gene expression profiles of different runs. After perform the mapping between runs and samples with run information table (generated from step (1)), gene expression profile for each sample can be generated.</h4>
    <ol>
        <li>Gene expression profiles of different runs (for internal use only).</li>
        <li>Gene expression profiles of different samples after performing the mapping between runs and samples with run information table (for internal use only).</li>
        <li>Gene expression profile table: A table in csv format contains gene expression profiles of all samples you are interested in after concatenating all gene expression profiles of different samples.<br>
            Each row represent different genes and each column represent different samples.
        </li>
    </ol>
    
### 5. Data normalization
<h4>This step will take the gene expression profile table as input and perform normalization to reduce the error across different studies.<br></h4>
    <ul>
        <li>Gene expression profile table: Generated from step (4). A table in csv format contains gene expression profiles of all samples you are interested in</li>
    </ul>
<h4>The output file are the normalized gene expression profile table. In addition, a binary file recorded the normalized gene expression table and all parameters are saved.<br></h4>
    <ol>
        <li>Normalized gene expression profile table: A table in csv format contains normalized gene expression profiles of all samples.</li>
        <li>Compendium saved in binary format: A python object contains the normalized gene expression table and recorded parameters. It can be used for the next step -- optional validation.</li>
    </ol>
    
### 6. Optional validation (Please read the document about validation and step-by-step examples for more information.)
<h4>This optional step will estimate the quality of normalized gene expression table. Unsupervised approaches and supervised approaches are provided.</h4>

![Figure 1. The entire transcriptomic compendium pipeline](https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/images/Figure1.png)
Figure 1. The entire transcriptomic compendium pipeline



## Configuration before using
<h4> To use this pipeline, you have to download the entire directory, install all required software and python packages. <h4>

### Download the entire directory
Please clone the entire repository into your computer.

### Install all required software
Required software have to be installed.

#### 1. Python 3.6 or later (NOTE: This version does not support Python 2 anymore)
<h4>Please download and install python 3 by following the instructions in official website:<br>
( https://www.python.org/downloads/ )</h4>
    <ul>
        <li>Tested in Python 3.6 </li>
    </ul>
    
#### 2. sratoolkit 2.9.6 or later
<h4>Please download and install sratoolkit by following the instructions in official website:<br>
( https://ncbi.github.io/sra-tools/install_config.html )</h4>
    <ul>
        <li>Tested with sratoolkit 2.9.6 </li>
        <li>Please make sure the toolkit location is specified in your $PATH variable. To add the toolkit location:
            <ul>
                <li>Linux: Use export command:
                
                    
    export PATH=<your_toolkit_location>:$PATH
                    
                    
</li>
                
<li>Windows 10: please follow these steps:
        <ol>
            <li>Open "Control Panel".</li>
            <li>Click "System".</li>
            <li>Click "About", and then click "System info" at the bottom of the page. </li>
            <li>Click "Advanced system settings" at the right of the system info page. </li>
            <li>Click "Environment Variables..." Button in advanced system settings window. </li>
            <li>Now you can add the toolkit location to PATH variable in Windows 10.</li>
        </ol>
</li>
</ul>
</li>
</ul>

#### 3. Bowtie 2.3.4 or later
<h4>Please download and install Bowtie2 by following the instructions in official website:<br>
( http://bowtie-bio.sourceforge.net/bowtie2/index.shtml )</h4>
    <ul>
        <li>Tested with Bowtie 2.3.4</li>
        <li>Please make sure the toolkit location is specified in your $PATH variable. (As installing sratoolkit)</li>
    </ul>

### Install all required Python packages
Required python packages have to be installed.

#### 1. biopython
<h4>Please install biopython by following the instructions in official website:<br>
( https://biopython.org/wiki/Download )</h4>
    <ul>
        <li>Tested with biopython 1.74</li>
    </ul>

#### 2. pandas
<h4>Please install pandas by following the instructions in official website:<br>
( https://pandas.pydata.org/docs/getting_started/install.html#installing-pandas )</h4>
    <ul>
        <li>Tested with pandas 0.25.0</li>
    </ul>
    
#### 3. RSeQC
<h4>Please install RSeQC by following the instructions in official website:<br>
( http://rseqc.sourceforge.net/#download-rseqc )</h4>
    <ul>
        <li>Tested with RSeQC 3.0.0</li>
        <li>Please make sure the location of 'infer_experiment.py' is specified in your $PATH variable. (As installing sratoolkit)</li>
            <ul>
                <li>(Example location: '~/.local/bin')</li>
            </ul>
    </ul>
    
#### 4. HTSeq
<h4>Please install HTSeq by following the instructions in official website:<br>
( https://htseq.readthedocs.io/en/master/ )</h4>
    <ul>
        <li>Tested with HTSeq 0.11.2</li>
        <li>Please make sure the location of 'htseq-count' is specified in your $PATH variable. (As installing sratoolkit)</li>
            <ul>
                <li>(Example location: '~/.local/bin')</li>
            </ul>
    </ul>
    
#### 5. missingpy
<h4>Please install missingpy by following the instructions in official website:<br>
( https://pypi.org/project/missingpy/ )</h4>
    <ul>
        <li>Tested with missingpy 0.2.0</li>
    </ul>
    
### Testing after install and configure your computer
<h4>After you added these path to PATH variable, you should be capable to run the following program in any directory:</h4>
<ul>
    <li>prefetch</li>
    <li>bowtie2</li>
    <li>infer_experiment.py</li>
    <li>htseq-count</li>
</ul>
<h4>If you failed to run these four programs, please make sure that you located these four programs correctly and added the correct path to PATH variables before you run this pipeline.</h4>

    
## How to use it
<h4>Please refer <a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/STEP-BY-STEP.md>step-by-step example</a> for more information</h4>

### Building compendium script (Main script) (build_compendium_script.py)
<h4>This script is the main script and will take sample list and a gene annotation file as inputs to build the compendium</h4>

#### Input
This script takes two input files and one additional argument:
<ul>
    <li> Input files:
        <ol>
            <li>Sample List: A file in csv format with one column with name "Experiment". (Please refer figure 1).(<a href="https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/SalmonellaExampleSampleList.csv">Example</a>)</li>
            <li>Gene Annotation: A file in gff3 format downloaded from NCBI genome database. (Please refer figure 1)(<a href="ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/006/945/GCA_000006945.2_ASM694v2/GCA_000006945.2_ASM694v2_genomic.gff.gz">Example (need decompression)</a>)</li>
        </ol>
    </li>
    <li> One additional argument:
        <ul>
            <li>Compendium name: The name you want. The pipeline will create a directory with this name and store all results in this directory.</li>
        </ul>
    </li>
</ul>

#### Output
This script will generate a directory with specified compendium name and many files in the directory. There are two the most important files:
<ul>
    <li>Normalized Data Matrix (Filename: '(Compendium Name)_NormalizedDataMatrix.csv'): A table in csv format contains normalized gene expression profiles of all samples. Each row represent different genes and each column represent different samples. (Please refer step (5) and figure 1)</li>
    <li>Compendium saved in binary format (Filename: '(Compendium Name)_projectfile.bin'): A python object contains the normalized gene expression table and recorded parameters. It can be used for optional validation. (Please refer step (5) and figure 1)</li>
</ul>

<a href = https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/raw/Pipeline_20200307/SalmonellaExample.tar.gz>Salmonella compendium example including validation results (need decompression)</a>
#### Usage

```
python build_compendium_script.py <sample list file path> <gene annotation file path> <compendium name>
```

### Unsupervised validation script (unsupervised_validation_script.py)
<h4>Please refer <a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/STEP-BY-STEP.md>step-by-step</a> example and validation description for more information</h4>

#### Input
This script takes only one input: Your compendium name.

#### Output
This script will generate two files in your compendium directory. There is also a benchmark shown in command line
<ul>
    <li> Output files:
        <ol>
            <li>Unsupervised validation result table (Filename: '(Compendium Name)_UnsupervisedValidationResults.csv'): A table recorded the symmetric mean absolute percentage error for different combination of noise ratio and missing value ratio.</li>
            <li>Unsupervised validation result figure (Filename: '(Compendium Name)_UnsupervisedValidationResults.png'): Visualization of the table.
            
![Figure 2. Unsupervised validation results of Salmonella Example compendium](https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/images/SalmonellaExample_UnsupervisedValidationResults.png)
<br>Figure 2. Unsupervised validation results of Salmonella Example compendium.
            
</li>
</ol>
</li>
<li> Bench mark shown in command line: A number between 0%~100%. The higher the number, the better the compendium. 
<ul>
<li>For this example, the number is about 50% (it varies due to the randomless).</li>
<li><a href=https://www.sciencedirect.com/science/article/pii/S1931312813004113">One small (26 samples), published Salmonella compendium</a> can be used for comparison. For that compendium, the benchmark is about 55%.</li>
<li>If we pick these 26 samples from Salmonella example compendium (709 samples) and perform this analysis, the benchmark is about 58%</li>
</ul>
</li>
</ul>

#### Usage

```
python unsupervised_validation_script.py  <compendium name>
```


### Unsupervised validation script (csv input file and user specified output) (unsupervised_validation_script_csv_input.py)
<h4>Please refer <a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/STEP-BY-STEP.md>step-by-step</a> example and validation description for more information</h4>

This script is similar as unsupervised validation script but it takes three more input. It will read external data matrix file and users have to specified the output file and figure name. <br>
This script is useful when you want to compare the benchmark of your compendium with external data matrix such as published data.
#### Input
This script takes three inputs in addition to the compendium name.
<ol>
    <li>Data matrix file: A csv files that contains gene expression profiles with arbitrary #rows and #columns. The first column will be assumed as gene name and the first row will be assumed as sample name.</li>
    <li>Output result table path</li>
    <li>Output result figure path</li>
</ol>

#### Output
This script will generate two files same as regular unsupervised validation script with user specified filename. There is also a benchmark shown in command line


#### Usage

```
python unsupervised_validation_script_csv_input.py  <compendium name> <output result table path> <output result figure path>
```


### Supervied validation (Correlation) script (supervised_validation_corr_script.py)
<h4>Please refer <a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/STEP-BY-STEP.md>step-by-step</a> example and validation description for more information</h4>

#### Input
This script takes two inputs: Your compendium name and sample-study-condition mapping table.
<ul>
    <li>Sample-study-condition mapping table: A csv file contains study name (or ID) and condition for selected samples in your compendium (<a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/SalmonellaExample_CorrelationValidation.csv>Example</a>). It contains three columns:
    <ol>
        <li>exp_id: Sample ID. (should be a subset of the sample list you provided when you prepared the compendium)</li>
        <li>series_id: Study ID or Study Name. </li>
        <li>cond_id: Condition name.</li>
    </ol>
    </li>
    
</ul>

#### Output
This script will generate two files in your compendium directory. 
<ul>
    <li> Output files:
        <ol>
            <li>Correlation validation result table (Filename: '(Compendium Name)_CorrelationValidationResults.csv'): A table recorded the average correlations among different samples or across different studies/conditions.</li>
            <li>Correlation validation result figure (Filename: '(Compendium Name)_CorrelationValidationResults.png'): Visualization of the table.
            
![Figure 3. Correlation validation results of Salmonella Example compendium](https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/images/SalmonellaExample_CorrelationValidationResults.png)
<br>Figure 3. Correlation validation results of Salmonella Example compendium.
            
</li>
</ol>
</li>
</ul>

#### Usage

```
python supervised_validation_corr_script.py  <compendium name> <sample-study-condition mapping file path>
```


### Supervied validation (Knowledge capture) script (supervised_validation_knowledge_capture_script.py)
<h4>Please refer <a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/STEP-BY-STEP.md>step-by-step</a> example and validation description for more information</h4>

#### Input
This script takes three inputs: Your compendium name, sample selection table and gene selection table.
<ul>
    <li>Sample selection table: A csv file contains samples and case/control indicator(<a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/Input_KnowledgeCapture_fur.csv>Example</a>). It contains two columns:
    <ol>
        <li>exp_id: Sample ID. (should be a subset of the sample list you provided when you prepared the compendium)</li>
        <li>indicator: inticate case or control (1 means case and 0 means control)</li>
    </ol>
    </li>
    <li>Gene selection table: A csv file contains genes(<a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/Input_KnowledgeCapture_fur_related_genes.csv>Example</a>). It contains one column:
    <ol>
        <li>gene_list: Gene names. (should be a subset of gene names in the normalized data matrix.)</li>
        <li>Other columns are just for comment and will not be processed.</li>
    </ol>
    </li>
</ul>

#### Output
This script will generate two files in your compendium directory. 
<ul>
    <li> Output files:
        <ol>
            <li>Knowledge Capture validation result table (Filename: '(Compendium Name)_KnowledgeCaptureValidationResults.csv'): A table recorded the sorted rank of absolute log fold change between case and control of selected genes for different noise ratio.</li>
            <li>Knowledge Capture validation result figure (Filename: '(Compendium Name)_KnowledgeCaptureValidationResults.png'): Visualization of the table.
            
![Figure 4. Knowledge capture validation results of Salmonella Example compendium](https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/images/SalmonellaExample_KnowledgeCaptureValidationResults.png)
<br>Figure 4. Knowledge capture validation results of Salmonella Example compendium.
            
</li>
</ol>
</li>
</ul>

#### Usage

```
python supervised_validation_knowledge_capture_script.py  <compendium name> <Sample selection table> <Gene selection table>
```


### Supervied validation (Published data comparison) script (supervised_validation_published_data_comparison_script.py)
<h4>Please refer <a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/STEP-BY-STEP.md>step-by-step example </a> and validation description for more information</h4>

#### Input
This script takes two input: Your compendium name and published data table.
<ul>
    <li>Published data table: A csv file contains published gene expression profiles.(<a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/SampleTable_STM_GoldenStandard.csv>Example</a>).
    <ol>
        <li>Each row represent a gene. The gene name should be consistent with the compendium.</li>
        <li>Each column represent a sample. The sample should be in the compendium.</li>
        <ul>
            <li>Some published data may take the average on technical replicates. In this case, you can concatenate these technical replicates with '|' symbol. The script will identify these replicates and merge them into one gene expression profile by taking an average (see the example file).</li>
        </ul>
    </ol>
    </li>
</ul>

#### Output
This script will generate two files in your compendium directory. 
<ul>
    <li> Output files:
        <ol>
            <li>Published data comparison result table (Filename: 'PublishedDataComparisonResults.csv'): A table recorded the log2 gene expression profiles from both compendium and published data.</li>
            <ul>
                <li>Only the genes recorded in both published data and compendium will be shown.</li>
            </ul>
            <li>Published data comparison result figure (Filename: 'PublishedDataComparisonResults.png'): Visualization of the table.
            
![Figure 5. Published data comparison results of Salmonella Example compendium](https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/images/PublishedDataComparisonResults.png)
<br>Figure 5. Published data comparison results results of Salmonella Example compendium.
            
</li>
</ol>
</li>
</ul>

#### Usage

```
python supervised_validation_published_data_comparison_script.py  <compendium name> <published data table path>
```


