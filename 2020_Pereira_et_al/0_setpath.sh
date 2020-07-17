#!/bin/bash
home_p=$(dirname $PWD)

raw_seq_p=$home_p/raw_seq

index_p=$home_p/index
single_p=$home_p/single

bowtie2=/usr/bin/bowtie2
index_bowtie2=/home/wxk/ref/ecoli/MS/Bowtie2Index/e_coli
rRNA_p=/home/wxk/ref/rRNA
rmrRNA_p=$home_p/rmrRNA

trim_p=$home_p/trim
sam_p=$home_p/sam
sam_rock_p=$home_p/sam_rock
count_p=$home_p/count
large10_p=$home_p/large10
condition_p=$home_p/condition
adapter_p=/home/wxk/software/Trimmomatic-0.33/adapters
trim_s_p=/usr/share/java/trimmomatic.jar
tophat_s=/usr/bin/tophat2
ref_p=/home/wxk/ref/ecoli
rock_ref_p=/home/wxk/ref/rock_ref
rock_s=/home/wxk/software/rockhopper/Rockhopper.jar
thout_p=$home_p/thout
htseq_s=/home/wxk/.local/bin/htseq-count
samtools_s=/usr/bin/samtools 
summary_p=$home_p/summary
