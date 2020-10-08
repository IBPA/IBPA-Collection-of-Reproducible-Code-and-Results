
# AutomatedOmicsCompendiumPreparationPipeline
This toolkit can prepare the compendium by collecting the samples in <a href="https://www.ncbi.nlm.nih.gov/sra">Sequencing Read Archive (SRA) </a> database.
(In the future, this toolkit will be capable to process microarray dataset from GEO and ArrayExpress database)

<h1> How to use it </h1>
<ol>
<li> Download all necessary software and toolkit </li>
<li> Add all installed software to PATH environment variables </li>
<li> Download the entire directory of this project </li>


</ol>


<h3>Required Software</h3>
<ol>
<li>Python3 (>=3.6.9) or Python2 (>=2.7.1)</li>
<li>sratoolkit (>=2.9.6) </li>
</ol>

<h3>Required Packages</h3>
<ol>
<li>biopython (>=1.74)</li>
<li>pandas (>=0.25.0)</li>
<li>RSeQC (For Python3: >=3.0.0, For Python2: 2.6.4)</li>
<li>HTSeq (>=0.11.2) </li>
</ol>


#
<h3>Update</h3>
<ul>
<li>08/17/2019 sequencing pipeline support parallel in local machine or cluster (SLURM)
<li>07/24/2019 implementation of fetching SRA data and convert SRA data to fastq file </li>
<li>07/24/2019 Use more set/get methods, wrapper of bowtie2-build, and compatability of the toolkit (sratoolkit) and packages (RSeQC and HTSeq) </li>
<li>07/23/2019 Gene Annotation Data Definition and SRA metadata download implementation</li>
<li>07/22/2019 01:43 Update: Metadata definitions</li>
<li>07/18/2019 21:20 Update: (TEST) Create the directories and files</li>
<li>07/18/2019 Update: Create the new repository</li>

</ul>



<h4>08/17/2019 Update</h4>
<ol>
<li>Sequencing Pipeline (Parallel Version) : (Finished functional part, need to implement API including parameters settings)</li>
<li>Data Postprocessing<br>
    <ol>
    <li>Concatenation (DONE)</li>
    <li>Imputation<br>
        <ol>
        <li>Missing Forest (Finished function part including parallel version, need to implement API and performance evaluation) </li>
        </ol>
    </li>
    <li>Normalization (ONGOING)</li>
    </ol>
</li>

<li>Data Validation<br>    
    <ol>
    <li>Unsupervised data validation (Will start after data postprocessing)</li>
    <li>Supervised data validation (Will start after data postprocessing)<br>
    </li>
</li>
</ol>