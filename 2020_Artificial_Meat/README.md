# A Comprehensive Sensitivity Analysis for the Animal-cell-based Meat Predictor

## Introduction

This project aims to provide a comprehensive sensitivity analysis for Animal-cell-based Meat (ACBM) model [1]. This folder only contains analysis results of the analysis. The codes will be released when the manuscript is published. Please send email to fzli@ucdavis.edu if you would like to review the code!

## Methods

![fig](/2020_Artificial_Meat/fig/spiderplot.png)

The analysis consists of the application of 6 algorithms: Morris Method (MM), Sobol Sensitivity Analysis (SSA), Random Balance Designs Fourier Amplitude Sensitivity Test (RBD-FAST), Fourier Amplitude Sensitivity Test (FAST), Delta Moment-Independent Measure (DMIM), and Derivative-based Global Sensitivity Measure (DGSM).

The model contains 67 parameters, and for each SA algorithm, the analyzer recorded 5 parameters with the most sensitivity measures. After eliminating the duplicates among these 30 parameters, there were 9 parameters having high sensitivity measures. We used these 9 parameter to plot the figure above.

|          | rho_c    | V_c      | conc_glu | GCR_c    | C_fgf2   | conc_fgf2 | t_m      | conc_tgfb | OUR_c    |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| DGSM     | 6.83E-03 | 1.00E+00 | 2.70E-02 | 5.70E-01 | 2.40E-03 | 5.07E-02  | 8.03E-03 | 4.93E-02  | 8.68E-02 |
| SSA      | 1.00E+00 | 9.66E-01 | 9.48E-01 | 8.80E-01 | 8.50E-01 | 7.47E-01  | 6.95E-01 | 2.16E-03  | 1.69E-03 |
| DMIM     | 8.90E-01 | 1.00E+00 | 9.47E-01 | 7.58E-01 | 7.83E-01 | 9.10E-01  | 5.98E-01 | 1.37E-02  | 5.13E-02 |
| FAST     | 7.82E-01 | 1.00E+00 | 5.83E-01 | 8.63E-01 | 4.97E-01 | 8.50E-01  | 6.94E-01 | 1.59E-04  | 1.93E-06 |
| MM       | 1.00E+00 | 9.70E-01 | 9.91E-01 | 9.53E-01 | 9.11E-01 | 9.09E-01  | 8.62E-01 | 1.44E-02  | 1.22E-08 |
| RBD-FAST | 1.00E+00 | 7.94E-01 | 9.96E-01 | 7.54E-01 | 7.86E-01 | 7.11E-01  | 8.22E-01 | 1.39E-01  | 7.48E-02 |

This table contains the detailed sensitivity measures of the analysis.

## Authors

- Fangzhou Li - https://github.com/fangzhouli

## Ackowledgements

- Derrick Risner, for providing ACBM model and data
- Ilias Tagkopoulos, for advisory and documentation review
- SALib, for providing analysis tools https://salib.readthedocs.io/en/latest/

## References

- [1] Derrick Risner, et al. "Techno-economic assessment of animal cell-based meat." (2020)
