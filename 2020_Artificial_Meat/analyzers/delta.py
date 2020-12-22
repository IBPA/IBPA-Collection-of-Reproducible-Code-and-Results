# -*- coding: utf-8 -*-
"""delta.py description.

This module defines a subclass for Analyzer that provides sensitivity analysis
by Delta Moment-Independent Measure.

Author:
    Fangzhou Li - https://github.com/fangzhouli

"""

import pickle
from SALib.sample import latin
from SALib.analyze import delta
from ._base import Analyzer
from function import ACBM


class DeltaAnalyzer(Analyzer):
    """Analyzer subclass.

    Attributes:
        n_samples (int): The amount of sampling.
        seed_sample (int): The seed of sampling random generator.
        seed_analyze (int): The seed of analyzer random generator.

    """

    def __init__(self, n_samples=1000, seed_sample=9, seed_analyze=10):
        super().__init__()
        self.n_samples = n_samples
        self.seed_sample = seed_sample
        self.seed_analyze = seed_analyze

    def analyze(self):
        """Initiate the analysis, and stores the result at data directory.

        Generates:
            Analysis result at 'acbm/data/output/delta.txt'.

        """

        X = latin.sample(self.problem, self.n_samples, seed=self.seed_sample)
        Y = ACBM.evaluate(X)
        si = delta.analyze(self.problem, X, Y, seed=self.seed_analyze)
        pickle.dump(si, open(self.path_output + 'delta.txt', 'wb'))
