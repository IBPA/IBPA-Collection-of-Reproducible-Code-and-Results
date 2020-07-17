#!/bin/bash
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-cpu 2000
#SBATCH -t 2:00:00
#SBATCH -o output/slurm.%N.%j.out
#SBATCH -e output/slurm.%N.%j.err

#SBATCH -e output/slurm.%N.%j.err

source 0_setpath.sh
srun featureCounts  -t gene -g gene_id -a $ref_p/MS/e_coli.gtf -o $count_p/XID_12H_1.txt $large10_p/XID_12H_1.sam
srun tail -n +3 $count_p/XID_12H_1.txt > $count_p/t1XID_12H_1t1
srun awk '{print $1, $7}' $count_p/t1XID_12H_1t1 > $condition_p/XID_12H_1.txt


