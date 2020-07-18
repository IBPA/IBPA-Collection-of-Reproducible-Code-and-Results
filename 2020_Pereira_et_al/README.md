Tis repository contains the code for analyzing RNA-seq.

###How to run?
Install the softwares needed for the analysis, bowtie2, trimmomatic, featureCounts. 

The pipeline consists of 5 steps which are run sequentially. The pipeline is run on a cluster which uses slurm workload manager. To run the kth step, run sbatch k_submit.sh

The final output of each sample is a table in which each row represents a gene and each column denotes the number of counts mapped to each gene in a biological replicate.
