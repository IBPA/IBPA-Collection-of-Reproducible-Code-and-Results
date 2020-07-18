#!/bin/bash
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-cpu 7000
#SBATCH -t 4:00:00
#SBATCH -o output/slurm.%N.%j.out
#SBATCH -e output/slurm.%N.%j.err

source 0_setpath.sh
srun bowtie2 -x $index_bowtie2 -q $trim_p/XID_12H_1.trim.fastq   -S $sam_p/XID_12H_1.sam
