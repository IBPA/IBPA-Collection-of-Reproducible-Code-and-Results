# -*- coding: utf-8 -*-
"""dgsm.py description.

This module defines a subclass for Analyzer that provides sensitivity analysis
by Derivative-based Global Sensitivity Measure (DGSM).

Author:
    Fangzhou Li - https://github.com/fangzhouli

"""

import pickle
from SALib.sample import finite_diff
from SALib.analyze import dgsm
from ._base import Analyzer
from function import ACBM


class DGSMAnalyzer(Analyzer):
    """Analyzer subclass.

    Attributes:
        n_samples (int): The amount of sampling.
        seed_sample (int): The seed of sampling random generator.
        seed_analyze (int): The seed of analyzer random generator.
        delta (float): The step size of differential equation.

    """

    def __init__(self, n_samples=1000, seed_sample=11, seed_analyze=12,
                 delta=0.0001):
        super().__init__()
        self.n_samples = n_samples
        self.seed_sample = seed_sample
        self.seed_analyze = seed_analyze
        self.delta = delta

    def analyze(self):
        """Initiate the analysis, and stores the result at data directory.

        Generates:
            Analysis result at 'acbm/data/output/dgsm.txt'.

        """

        X = finite_diff.sample(self.problem, self.n_samples, delta=self.delta,
                               seed=self.seed_sample)
        Y = ACBM.evaluate(X)
        si = dgsm.analyze(self.problem, X, Y, seed=self.seed_analyze)

        # scale down the values of vi
        si['vi'] = [x ** (1 / 16) for x in si['vi']]
        pickle.dump(si, open(self.path_output + 'dgsm.txt', 'wb'))
