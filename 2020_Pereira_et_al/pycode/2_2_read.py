#!/usr/bin/python

from read_path import *

import sys
import os
import gzip

map_file=open(index_p+'/'+sys.argv[1]+'.index.new','r')
read_file=gzip.open(raw_seq_p+'/R1.fastq.gz','rb')
out=open(single_p+'/'+sys.argv[1]+'.single.fastq','w')

index={}
linenum=0
i=0

for line in map_file.xreadlines():
        raw_num=int(line[:-1])
        transformed_num=(raw_num-2)/4+1
	index[transformed_num]=1

for line in read_file:
        line=line[:-1]
        if i%4 == 0:
                linenum=linenum+1
        i=i+1
        try:
                v=index[linenum]
                out.write(line+'\n')
        except:
                continue	

