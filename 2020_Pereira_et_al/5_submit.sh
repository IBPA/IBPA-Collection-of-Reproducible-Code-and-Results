#!/bin/bash
while read barcode index
do 
  sed -i "s/large10_p\/.*sam/large10_p\/$index\.sam/g" 5_featureCounts.sh
  perl -p -i -e "s/count_p\/.*?txt \$large/count_p\/${index}\.txt \$large/g"  5_featureCounts.sh
  perl -p -i -e "s/count_p\/.*?txt >/count_p\/${index}\.txt >/g"  5_featureCounts.sh
  perl -p -i -e "s/count_p\/t1.*?t1/count_p\/t1${index}t1/g" 5_featureCounts.sh
  perl -p -i -e "s/condition_p\/.*?txt/condition_p\/${index}\.txt/g"  5_featureCounts.sh
  sbatch 5_featureCounts.sh
done < map.txt
cd ./output
rm -f ./*
