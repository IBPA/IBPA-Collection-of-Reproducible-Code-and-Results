#!/bin/bash
#SBATCH -p med
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-cpu 2000
#SBATCH -t 3:00:00
#SBATCH -o output/slurm.%N.%j.out
#SBATCH -e output/slurm.%N.%j.err

srun -n 1 python ./pycode/2_2_read.py XID_12H_1
