# Adaptive laboratory evolution for improved tolerance of isobutyl acetate in _Escherichia coli_

This folder contains the code used to identify mutations described in [Matson et al., Metabolic Engineering (2021)](https://www.sciencedirect.com/science/article/pii/S1096717621001737)

* <b>breseq_code.py</b>: contains all code used to generate mutations. The commented lines at the top of the page describe how the reference sequence was derived from BW25113.gbk <br />

### Dependencies
* [breseq](https://barricklab.org/twiki/pub/Lab/ToolsBacterialGenomeResequencing/documentation/installation.html)
* python 3.6 or above

### Running
* Step1: Acquire all 25 early mutant fastq files, the ALE16 parent fastq file, the BW25113.gbk file, the K12 plasmid gbk file, and the placIq-LacI fasta file and place them in the same folder as breseq_code.py.

* Step2: Generate a new reference sequence by aligning the ALE16 parent fastq file to the BW25113.gbk, then adding those changes to BW25113.gbk as follows,

    * a. ```breseq -j 8 -o ALE17_ref_output -r BW25113.gbk parent_reseq.fastq```
    * b. ```gdtools APPLY -f GFF3 -o parent_reseq.gff3 -r BW25113.gbk parent_reseq.gd```
    * c. ```gdtools VALIDATE -r BW25113.gbk parent_reseq_DEL.gd```

* Step3: Run ```breseq_code.py```

* Step4: To locate the mutations, look at the files ```index.html``` found at 'output_folder'/output/index.html.

### Support

If you have any questions about this project, please contact Erol Kavvas (eskavvas@ucdavis.edu).