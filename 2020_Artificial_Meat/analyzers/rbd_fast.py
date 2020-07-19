# -*- coding: utf-8 -*-
"""rbd_fast.py description.

This module defines a subclass for Analyzer that provides sensitivity analysis
by Random Balance Designs - Fourier Amplitude Sensitivity Test (RBD-FAST).

Author:
    Fangzhou Li - https://github.com/fangzhouli

"""

import pickle
from SALib.sample import latin
from SALib.analyze import rbd_fast
from ._base import Analyzer
from function import ACBM


class RDBFastAnalyzer(Analyzer):
    """Analyzer subclass.

    Attributes:
        n_samples (int): The amount of sampling.
        seed_sample (int): The seed of sampling random generator.
        seed_analyze (int): The seed of analyzer random generator.
        M (int): The interference number.

    """

    def __init__(self, n_samples=1000, seed_sample=5, seed_analyze=6,
                 M=10):
        super().__init__()
        self.n_samples = n_samples
        self.seed_sample = seed_sample
        self.seed_analyze = seed_analyze
        self.M = M

    def analyze(self):
        """Initiate the analysis, and stores the result at data directory.

        Generates:
            Analysis result at 'acbm/data/output/rbd_fast.txt'.

        """

        X = latin.sample(self.problem, self.n_samples, seed=self.seed_sample)
        Y = ACBM.evaluate(X)
        si = rbd_fast.analyze(self.problem, X, Y, M=self.M,
                              seed=self.seed_analyze)
        pickle.dump(si, open(self.path_output + 'rbd_fast.txt', 'wb'))
