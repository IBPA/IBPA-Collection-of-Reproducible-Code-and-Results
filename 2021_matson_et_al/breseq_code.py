import os
import subprocess

## 1. Alignment of parent reads against BW25113 reference
# breseq -j 8 -o ALE17_ref_output -r BW25113.gbk parent_reseq.fastq

## 2. Apply the mutations described in the input GenomeDiff to the reference sequence in order to create new reference sequence
# gdtools APPLY -f GFF3 -o parent_reseq.gff3 -r BW25113.gbk parent_reseq.gd
# gdtools APPLY -f GFF3 -o parent_reseq_DEL.gff3 -r BW25113.gbk parent_reseq_DEL.gd

## 3. validate addition of new mutations
# gdtools VALIDATE -r BW25113.gbk parent_reseq_DEL.gd

### Mutations identified against parent strain
ref_id = "parent_reseq_DEL"
for i in range(1,24):
	mutant_id = "EM"+str(i) #+".fastq"
	print(mutant_id)

	cmd_str = "breseq -j 8 -o %s_parent_DEL_output -r %s.gff3 %s.fastq"%(mutant_id,ref_id,mutant_id)
	print(cmd_str)
	subprocess.call(cmd_str, shell=True)
	# subprocess.Popen(["breseq", "-j", "8", "-o", ""])


### Mutations identified against K12-plasmid
ref_id = "k12-plasmid"
for i in range(1,24): #range(18,24):
	mutant_id = "EM"+str(i) #+".fastq"
	print(mutant_id)

	cmd_str = "breseq -j 8 -o %s_k12plasmid_output -r %s.gbk %s.fastq"%(mutant_id,ref_id,mutant_id)
	print(cmd_str)
	subprocess.call(cmd_str, shell=True)
	# subprocess.Popen(["breseq", "-j", "8", "-o", ""])


### Mutations identified against Lacq-lacI region
# breseq -j 8 -o parent_PlacIq-lacI_output -r Placlq-lacI.fasta parent_reseq.fastq
ref_id = "Placlq-lacI.fasta"
for i in range(1,24):
# 	if i!=5 and i!=10 and i!=12:
	mutant_id = "EM"+str(i) #+".fastq"
	print(mutant_id)

	cmd_str = "breseq -j 8 -o %s_Placlq-lacI_output -r %s %s.fastq"%(mutant_id,ref_id,mutant_id)
	print(cmd_str)
	subprocess.call(cmd_str, shell=True)
# 	# subprocess.Popen(["breseq", "-j", "8", "-o", ""])