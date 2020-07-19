# -*- coding: utf-8 -*-
"""fast.py description.

This module defines a subclass for Analyzer that provides sensitivity analysis
by Fourier Amplitude Sensitivity Test (FAST).

Author:
    Fangzhou Li - https://github.com/fangzhouli

"""

import pickle
from SALib.sample import fast_sampler
from SALib.analyze import fast
from ._base import Analyzer
from function import ACBM


class FastAnalyzer(Analyzer):
    """Analyzer subclass.

    Attributes:
        n_samples (int): The amount of sampling.
        seed_sample (int): The seed of sampling random generator.
        seed_analyze (int): The seed of analyzer random generator.
        M (int): The interference number.

    """

    def __init__(self, n_samples=1000, seed_sample=3, seed_analyze=4,
                 M=4):
        super().__init__()
        self.n_samples = n_samples
        self.seed_sample = seed_sample
        self.seed_analyze = seed_analyze
        self.M = M

    def analyze(self):
        """Initiate the analysis, and stores the result at data directory.

        Generates:
            Analysis result at 'acbm/data/output/fast.txt'.

        """

        X = fast_sampler.sample(self.problem, self.n_samples, M=self.M,
                                seed=self.seed_sample)
        Y = ACBM.evaluate(X)
        si = fast.analyze(self.problem, Y, M=self.M, seed=self.seed_analyze)
        pickle.dump(si, open(self.path_output + 'fast.txt', 'wb'))
