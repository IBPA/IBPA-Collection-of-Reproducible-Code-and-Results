#!/bin/bash
while read barcode index
do 
  sed -i "s/sam_p\/.*sam/sam_p\/$index\.sam/g" 4_bowtie2.sh
  sed -i "s/trim_p\/.*trim/trim_p\/$index\.trim/g" 4_bowtie2.sh
  sbatch 4_bowtie2.sh
done < map.txt
cd ./output
rm -f ./*
