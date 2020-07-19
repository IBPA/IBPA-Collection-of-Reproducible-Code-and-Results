# A Comprehensive Sensitivity Analysis for the Animal-cell-based Meat Predictor

## Introduction

This project aims to provide a comprehensive sensitivity analysis for animal-cell-based meat (ACBM) cost prediction model.

## Usage

### Installation

```console
git clone git@github.com:fangzhouli/ACBM-SA.git
cd path/to/ACBM-SA
pip install .
```

### Usage

```console
python analyze.py  # generates analysis result files in 'acbm/data/output'
python plot.py  # create a spider plot to visualize SA
```

## Methods

![fig](/fig/spiderplot.png)
The analysis consists of the application of 6 algorithms: Morris Method (MM), Sobol Sensitivity Analysis (SSA), Random Balance Designs Fourier Amplitude Sensitivity Test (RBD-FAST), Fourier Amplitude Sensitivity Test (FAST), Delta Moment-Independent Measure (DMIM), and Derivative-based Global Sensitivity Measure (DGSM).

The model contains 67 parameters, and for each SA algorithm, the analyzer recorded 5 parameters with the most sensitivity measures. After eliminating the duplicates among these 30 parameters, there were 9 parameters having high sensitivity measures. We used these 9 parameter to plot the figure above.

## Authors

- Fangzhou Li - https://github.com/fangzhouli

## Ackowledgements

- Derrick Risner, for providing ACBM model and data
- Ilias Tagkopoulos, for advisory and documentation review
- SALib, for providing analysis tools https://salib.readthedocs.io/en/latest/
