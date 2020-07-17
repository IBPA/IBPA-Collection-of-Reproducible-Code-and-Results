# Tax Project for the California Department of Tax and Fee Administration

This folder contains the code used to generate predictor (both classifier and regressor) results in Main Manuscript, as well as results obtained in [Supplementary Materials](https://github.com/IBPA/IBPA-Collection-of-Reproducible-Code-and-Results/tree/master/2020_Chan_et_al/Supplementary). Due to the confidentiality of the data, the results cannot be faithfully reproduced. However, it is our hope that this code can be used by others seeking to fulfill similar goals. 

![Figure 1](https://github.com/IBPA/IBPA-Collection-of-Reproducible-Code-and-Results/blob/master/2020_Chan_et_al/Figures/Figure1/Figure1.png)
*Figure 1. Overview of the audit prediction pipeline.*

* <b>predictor.py</b>: contains all code written for classifier and regressor <br />
* <b>Figures</b>: folder containing Figures and code to generate them <br />
* <b>Supplementary</b>: folder containing code used to generate results found in Supplementary Materials <br />

### Dependencies
* [tensorflow](https://github.com/tensorflow/tensorflow)
* python 3.6 or above

### Running
* Step1: prepare a directory containing your input files (with exact names):

  * ```Data.csv```: comma-delimeted file with columns as different tax features

* Step2: Create text file ```features.txt``` with 

  * list of return features: features relating to tax return data
  * list of registrtation features: features relating to registration data
  * target feature for classification
  * Example: return features (gross sales, taxable transactions, ...) registration features (e.g. Business Type, City ID, ...) output feature

* Step3: ```python predictor.py```

Upon completion, ```pred.csv``` will contain the predicted protein identification probabilities.

### Support

If you have any questions about this project, please contact Trevor Chan (tchchan@ucdavis.edu).

### Acknowledgement

This work was supported by the California Department of Tax and Fee Administration (CDTFA).
