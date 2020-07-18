#!/bin/bash
while read barcode index
do 
  sed -i "s/-n.*\$raw/-n ^$barcode \$raw/g" 1_index.sh
  sed -i "s/x_p\/.*.index/x_p\/$index.index/g" 1_index.sh
  sbatch 1_index.sh
done <map.txt

