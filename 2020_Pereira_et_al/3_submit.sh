#!/bin/bash
while read barcode index
do 
  sed -i "s/\/.*single/\/$index\.single/g" 3_trim.sh
  sed -i "s/trim_p\/.*trim/trim_p\/$index\.trim/g" 3_trim.sh
  sbatch 3_trim.sh
done < map.txt

