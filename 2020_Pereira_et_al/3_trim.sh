#!/bin/bash
#SBATCH -p med
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-cpu 2000
#SBATCH -t 3:00:00
#SBATCH -o output/slurm.%N.%j.out
#SBATCH -e output/slurm.%N.%j.err

source 0_setpath.sh

srun -n 1 java -jar $trim_s_p SE -phred33 $single_p/XID_12H_1.single.fastq $trim_p/XID_12H_1.trim.fastq ILLUMINACLIP:$adapter_p/TruSeq3-SE.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36
