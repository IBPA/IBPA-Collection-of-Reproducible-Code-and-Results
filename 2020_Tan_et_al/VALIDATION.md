# Validation of the compendium
<h4>Before you read this part, please make sure you read the <a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/tree/Pipeline_20200307>main documentation first.</a></h4>
<h4>Please read <a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/STEP-BY-STEP.md>step-by-step</a> example for more information. </h4>
It is important to validate and check the quality of the compendium. There are one supervised approach and three unsupervised approaches to check the quality or validate the compendium you built.

## An unsupervised approach -- Drop and impute values approach
<h4>Unsupervised approaches can evaluate one or more benchmarks of the compendium you built based on several assumptions. Users do not have to provide additional information to obtain the benchmark.</h4>

### Assumptions
<h4>Based on the following two assumtpions, drop and impute values approach can evaluate a benchmark to evaluate the quality of the compendium.</h4>
<ol>
    <li>A good compendium should capture the pattern of gene expression profiles. Therefore, even if some values are dropped, we still can recover these missing values by applying missing value imputation.</li>
    <li>If a compendium is perturbed, it is more difficult to recover the missing values and yield larger error.</li>
</ol>

### Steps
<h4>There are four steps for drop and impute values approach:</h4>
<ol>
    <li>Add the noise to the normalized data matrix in the compendium with different noise ratio.
        <ul>
            <li>The noise for adding is from random permutation results of normalized data matrix to make sure that the noise has the same distribution as the original data matrix.</li>
            <li>The noise ratio is a value between 0 and 1.
            <ul>
                <li>If noise ratio is 0, it means that there is no noise to be added.</li>
                <li>If noise ratio is 1, it means that the data matrix is totally perturbed by random permutation.</li>
                <li>If noise ratio is between 0 and 1 (0 < x < 1, x is noise ratio), it means that the data matrix will be generated from the linear combination of original data matrix and noise:<br><br>
                    x*(noise) + (1-x)*(original data matrix)
                </li>
            </ul>
        </ul>
    </li>
    <li>Randomly remove the values in the noise-added data matrix with different missing value ratio.</li>
    <li>Run the missing value imputation to fill the missing value back.<br>
        There are two approaches to impute the missing value in the pipeline:
        <ul>
            <li>K-nearest neighborhood (KNN)</li>
            <li>Missing Forest</li>
        </ul>
        The pipeline use missingpy to perform these two imputation approaches. Please refer <a href=https://pypi.org/project/missingpy/>the package description</a> for more information.
    </li>
    <li>Calculate the error between imputed values and the original values before dropping.
        <ul>
            <li>There will be a error table recording the error of values for each noise ratio. By taking the average, the pipeline can get the average error of this noise ratio.</li>
            <li>There will be a average error table recording the average error of different noise ratio for each missing value ratio. Each table can be plotted as a noise ratio vs. average error curve.</li>
        </ul>
    </li> 
</ol>

![Figure V1. Four steps of drop and impute values approach](https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/images/Figure2.png)
Figure V1. Four steps of drop and impute values approach

### Benchmark evaluation
<h4> After we get the average error curve, the next thing is the benchmark evaluation.</h4>
<ul>
    <li>We can pick a curve (for example, a curve from missing value ratio = 0.5) and then calculate the area below this curve.
        <ul>
            <li>The large area means bad compendium quality due to that high error of missing value imputation.</li>
        </ul>
    </li>
    <li>Or, we can calculate the area between the selected curve and the reference line (for example, a horizontal line with 100% error).
        <ul>
            <li>The large area means good compendium quality due to that low error of missing value imputation.</li>
        </ul>
    </li>
</ul>
<h4>Current pipeline use the second defination as the benchmark. Since the noise ratio should be between 0 and 1 and typically error will not exceed 100%, the area will be between 0% and 100%.</h4>

![Figure V2. Benchmark evaluation of drop and impute values approach](https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/images/Unsupervised_validation_description.png)
Figure V2. Benchmark evaluation of drop and impute values approach. The area between 100% error horizontal line and the error curve from missing value ratio = 0.5 (orange line) is evaluated as the benchmark.

### Benchmark comparison with reference compendium
After you got the benchmark with 49.35%, you want to know whether 49.35% is high enough to prove the compendium quality. To check whether this value is high enough, you need to fine one reference compendium with good quality and then evalute its benchmark and then make the comparison.

<h4>Reference compendium with good quality</h4>
For this Salmonella example compendium with 709 samples, it includes <a href=https://www.sciencedirect.com/science/article/pii/S1931312813004113">one small, published compendium</a> with 26 samples across different conditions and we can use this as a reference compendium.
<ul>
    <li>The benchmark of the Salmonella example compendium (with 709 samples) is about 49.35%. (it varies due to the randomness of validation procedures)</li>
    <li>The benchmark of the reference compendium (with 26 samples) is about 55.51%.</li>
    <li>The benchmark of the subset of Salmonella example compendium (with 26 samples in the reference compendium) is about 58.7%.</li>
</ul>
In conclusion, the benchmark of the subset of Salmonella example compendium is comparable with the benchmark of the refeernece compendium. <br>
You may also observed that for the high missing value ratio (0.99), the imputation error of the entire Salmonella example compendium is lower than the reference compendium or the subset compendium. It may implies that capture more samples can capture more information for recovering the missing values in this validation procedure.

![Figure V3. Benchmark comparison results](https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/images/Unsupervised_validation_comparison.png)
Figure V3. Benchmark comparison results. (A) The results of the entire Salmonella example compendium with 709 samples. (B) The results of the reference compendium. (C) The results of the subset of Salmonella example compendium contains the samples in the reference compendium.


## An Supervised approach -- Correlation validation
<h4>Supervised approaches need additional information from users. For correlation validation, it needs samples-studies-conditions mapping table.</h4>

### Assumptions
<h4>Based on the following three assumtpions, correlation validation can perform simple validation of the compendium. (Currently there is no benchmark to evaluate overall quality, just observation and simple validation)</h4>
<ol>
    <li>The average correlation among samples within the same condition should be higher than the average correlation among samples within the same study.</li>
    <li>The average correlation among samples within the same study should be higher than the average correlation among samples in the entire compendium.</li>
    <li>The average correlation among different conditions or different studies should not be too high. (Otherwise the compendium may have low diversity)</li>
</ol>

### Steps (Correlation validation)
<h4>There are four steps for correlation validation approach to check the average correlation among samples within the same condition, study or the entire compendium (Figure V4(A)):</h4>
<ol>
    <li>Add the noise to the normalized data matrix in the compendium with different noise ratio.
        <ul>
            <li>Same as the adding noise step in drop and impute values approach</li>
        </ul>
    </li>
    <li>Read the samples-studies-conditions mapping table and group the samples by conditions or studies.
        <ul>
            <li>It need one user input: Sample-study-condition mapping table, a csv file contains study name (or ID) and condition for selected samples in your compendium (<a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/SalmonellaExample_CorrelationValidation.csv>Example</a>). It contains three columns:
            <ol>
                <li>exp_id: Sample ID. (should be a subset of the sample list you provided when you prepared the compendium)</li>
                <li>series_id: Study ID or Study Name. </li>
                <li>cond_id: Condition name.</li>
            </ol>
            </li>
        </ul>
    </li>
    <li>For each group, evaluate the correlation matrix.
        <ul>
            <li>The group with just only one sample will be skipped.</li>
        </ul>
    </li>
    <li>Take the average values in lower half part of the correlation matrix from all groups.
        <ul>
            <li>One average correlation will be evaluated for each noise ratio and then a noise ratio vs. correlation curve can be plotted.</li>
            <li>Three lines with different grouping approaches (group by conditions, studies, or the entire compendium) will be plotted.</li>
        </ul>
    </li>
</ol>

### Steps (Correlation validation for checking diversity)
<h4>There are three steps for correlation validation approach to check the diversity among conditions and studies after adding the noise to normalized data matrix with different noise ratio. (Figure V4(B)):</h4>
<ol>
    <li>Read the samples-studies-conditions mapping table and group the samples by conditions or studies.
        <ul>
            <li>It need one user input: Sample-study-condition mapping table, a csv file contains study name (or ID) and condition for selected samples in your compendium (<a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/SalmonellaExample_CorrelationValidation.csv>Example</a>). It contains three columns:
            <ol>
                <li>exp_id: Sample ID. (should be a subset of the sample list you provided when you prepared the compendium)</li>
                <li>series_id: Study ID or Study Name. </li>
                <li>cond_id: Condition name.</li>
            </ol>
            </li>
        </ul>
    </li>
    <li>For each group, merge the gene expression by taking the average.
    </li>
    <li>Calculate the correlation matrix among groups and take the average of the lower half part of the correlation matrix.
        <ul>
            <li>One average correlation will be evaluated for each noise ratio and then a noise ratio vs. correlation curve can be plotted.</li>
            <li>Two lines with different grouping approaches (group by conditions and studies) will be plotted.</li>
        </ul>
    </li>
</ol>

### Evaluation and observation of the correlation validation results
<h4>There are several points to simply validate the compendium based on the results: (Figure V4)</h4>
<ol>
    <li>For correlation validation, the correlation curve from grouping by conditions (the green curve) should be higher than the curve from grouping by studies (the orange curve).</li>
    <li>For correlation validation, the correlation curve from grouping by studies (the orange curve) should be higher than the curve from the entire compendium (the blue curve).</li>
    <li>For checking diversity, the correlation curve among different studies or conditions (the red and purple curve) should not be too high. However, the exact threshold is not defined in the current version.</li>
</ol>

![Figure V4. Evaluation and observation of the correlation validation results.](https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/images/SalmonellaExample_CorrelationValidationResults.png)
<br>Figure V4. Evaluation and observation of the correlation validation results. It follow the first two criteria: the green curve (average correlation of samples grouping by conditions) is higher than orange curve (average correlation of samples grouping by studies), and the orange curve is higher than the blue line (average correlation of the entire compendium).


## An Supervised approach -- Knowledge capture validation
<h4>Supervised approaches need additional information from users. For knowledge capture validation, it needs two inputs:</h4>
<ol>
    <li>Sample selection table: A csv file contains samples and case/control indicator(<a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/Input_KnowledgeCapture_fur.csv>Example</a>). It contains two columns:
    <ol>
        <li>exp_id: Sample ID. (should be a subset of the sample list you provided when you prepared the compendium)</li>
        <li>indicator: inticate case or control (1 means case and 0 means control)</li>
    </ol>
    </li>
    <li>Gene selection table: A csv file contains genes(<a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/Input_KnowledgeCapture_fur_related_genes.csv>Example</a>). It contains one column:
    <ol>
        <li>gene_list: Gene names. (should be a subset of gene names in the normalized data matrix.)</li>
        <li>Other columns are just for comment and will not be processed.</li>
    </ol>
    </li>
</ol>

### Assumptions
<h4>A good compendium should capture the published information and knowledge. This validation focus on the fold change of average gene expression profiles between case and control. There are two levels of knowledge capture:</h4>
<ol>
    <li>Capture the information in specific study: If one published study with published gene expression profiles shows that some genes are up-regulated or down-regulated with significant fold change for specific stresses or mutants, the data matrix of the compendium should keep this information.</li>
    <li>Capture the general knowledge: If there are well known knowledge (or one published study without available gene expression profiles) points that some genes are up-regulated or down-regulated with significant fold change for specific stresses or mutants, the data matrix of the compendium should keep this information.</li>
</ol>

The second level is more difficult but also more valuable:

<ul>
    <li>For example, if study A points that some gene were up-regulated or down-regulated with significant fold change on some stresses or mutants.</li>
    <ul>
        <li>The compendium can show the signficant fold change on the same stress or the same mutant in different studies "in additiona to" study A.</li>
        <li>It means that we can capture more samples about this mutant or stress for the further study even those studies are not focus on this mutant and stress.</li>
    </ul>
</ul>

### Steps
<h4>There are four steps for knowledge capture validation approach (Figure V5):</h4>
<ol>
    <li>Add the noise to the normalized data matrix in the compendium with different noise ratio.
        <ul>
            <li>Same as the adding noise step in drop and impute values approach</li>
        </ul>
    </li>
    <li>Read the sample selection table and pick the case samples and control samples, and then calculate the absolute log fold change of average gene expressions between case samples and control samples. Then the rank of absolute log fold change is evaluated.
        <ul>
            <li>Sample selection table: A csv file contains samples and case/control indicator(<a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/Input_KnowledgeCapture_fur.csv>Example</a>). It contains two columns:
            <ol>
                <li>exp_id: Sample ID. (should be a subset of the sample list you provided when you prepared the compendium)</li>
                <li>indicator: inticate case or control (1 means case and 0 means control)</li>
            </ol>
            </li>
            <li>It calculate the absolute log fold change, which means that it does not consider the genes are up-regulated or down-regulated.</li>
        </ul>
    </li>
    <li>Read the gene selection table and pick the genes which is expected to have significant fold change. The rank of absolute log fold change of these gene are selected and sorted.</li>
        <ul>
            <li>Gene selection table: A csv file contains genes(<a href=https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/TestFiles/Input_KnowledgeCapture_fur_related_genes.csv>Example</a>). It contains one column:
            <ol>
                <li>gene_list: Gene names. (should be a subset of gene names in the normalized data matrix.)</li>
                <li>Other columns are just for comment and will not be processed.</li>
            </ol>
            </li>
        </ul>
    </li>
    <li>Count the gene one by one and then divided by the #genes to get hit ratio. Plot the rank vs. hit ratio for different noise ratio cases.
        <ul>
            <li>#genes is the number of genes in the gene list. (If some genes in the gene list are not in the compendium, then those genes will not be counted)</li>
        </ul>
    </li>
</ol>

![Figure V5. Knowledge capture validation steps.](https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/images/Figure4.png)
<br>Figure V5. Knowledge capture validation steps.

### Validation of the results
The rank of the absolute log fold change should be significantly different than random numbers draw from uniform distribution. Therefore, the sorted-rank of the absolute log fold change can be compared with a series of random number from uniform distribution by KS-test.
<ul>
    <li>Salmonella example compendium: There are two knowledge capture validation example results.
        <ol>
            <li>fur mutant case: There are one fur mutant and two wildtype samples for comparison. 
                <ul>
                    <li>fur mutant (case): SRX1638999 </li>
                    <li>wildtype (control): SRX1638996, SRX1638997 </li>
                    <li>Genes: according the the <a href=https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5001712/>published study</a>, it is well known that fur genes activates flagellar genes, SPI1 genes but represses SPI2 genes. Total are 26 genes.</li>
                    <li>Results (Figure V6(A)): The KS-test shows significant results (p-value < 0.05) for all cases with noise ratio < 0.5.
                </ul>
            </li>
            <li>hfq mutant case: There are 10 hfq mutant and 10 hfq samples from six different Salmonella strains for comparison. 
                <ul>
                    <li>hfq mutant (case): ERX339078,ERX339082,ERX339070,ERX339086,ERX339072,ERX339084,ERX339074,ERX339088,ERX339080,ERX339076 </li>
                    <li>wildtype (control): ERX339077,ERX339081,ERX339069,ERX339085,ERX339071,ERX339083,ERX339073,ERX339087,ERX339079,ERX339075 </li>
                    <li>Genes: according the the <a href=https://www.ncbi.nlm.nih.gov/pubmed/?term=23808344>published study</a>, it is well known that hfq gene regulated 9 genes. (7 genes are in the compendium)</li>
                    <li>Results (Figure V6(B)): The KS-test shows significant results (p-value < 0.05) for the case only without noise ratio.
                </ul>
            </li>
        </ol>
    </li>
    <li>Human example (Ischemic heart disease) compendium: There is one knowledge capture validation example result. (This does not show the general knowledge capture, just shows the information capture for the specific study)
        <ul>
            <li>Ischemic heart disease: There are 13 Ischemic heart patient and 14 healthy controls.
                <ul>
                    <li>patients (case): SRX4297657,SRX4297658,SRX4297659,SRX4297660,SRX4297661,SRX4297662,SRX4297663,SRX4297664,SRX4297665,SRX4297666,SRX4297667,SRX4297668,SRX4297669 </li>
                    <li>healthy (control): SRX4297606,SRX4297608,SRX4297610,SRX4297611,SRX4297612,SRX4297613,SRX4297614,SRX4297615,SRX4297616,SRX4297617,SRX4297618,SRX4297619,SRX4297609,SRX4297607 </li>
                    <li>Genes: according the the <a href=https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6233272/>this study </a>, there are 93 genes with absolute log fold change > 2. (64 genes are in the compendium)</li>
                    <li>Results (Figure V7): The KS-test shows significant results (p-value < 0.05) for all cases with noise ratio < 0.5.
                </ul>
            </li>
        </ul>
    </li>
</ul>

![Figure V6. Knowledge capture validation results--Salmonella.](https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/images/knowledge_capture_Salmonella.png)
<br>Figure V6. Knowledge capture validation results--Salmonella. (A) fur mutant case. (B) hfq mutant case.

![Figure V7. Validation of information capture in specific study--Human (Ischemic heart disease case).](https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/images/IschemicHeartDiseaseSimplified2_KnowledgeCaptureValidationResults.png)
<br>Figure V7. Validation of information capture in specific study--Human (Ischemic heart disease case).


## An Supervised approach -- Published data comparison
You can compare your compendium with the published data directly. The pipeline will extract the genes and samples in both your compendium and published data, and then calculate the correlation for each sample.

### Assumptions
<h4>A gene expression profile in a good compendium should have high similarity comapred with a gene expression profile in the published data.</h4>

### Steps
<h4>This approach has two steps:</h4>
<ol>
    <li>Read published data matrix and then extract the genes and samples in the compendium.</li>
    <li>Calculate the correlation between the gene expression profile in the compendium and given published data matrix for each sample, and then plot the log gene expression level.</li>
</ol>

![Figure V8. Published data comparison steps.](https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/images/Figure5.png)
<br>Figure V8. Published data comparison steps.

### Validation of the results:
Even without any additional normalization and standardization, the average PCC and SCC between published data and the compendium should be higher than 0.8. For the comparison case: Salmonella example compendium vs. <a href=https://www.sciencedirect.com/science/article/pii/S1931312813004113">published small compendium</a>, both average PCC and SCC are around 0.9.

![Figure V9. Published data comparison steps.](https://github.com/bigghost2054/AutomatedOmicsCompendiumPreparationPipeline/blob/Pipeline_20200307/images/PublishedDataComparisonResults.png)
<br>Figure V9. Published data comparison results (Salmonella example compared with published compendium ).
