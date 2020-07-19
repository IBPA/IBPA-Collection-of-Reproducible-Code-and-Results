# -*- coding: utf-8 -*-
"""morris.py description.

This module defines a subclass for Analyzer that provides sensitivity analysis
by Method of Morris.

Author:
    Fangzhou Li - https://github.com/fangzhouli

"""

import pickle
from SALib.sample.morris import sample as m_sample
from SALib.analyze.morris import analyze as m_analyze
from ._base import Analyzer
from function import ACBM


class MorrisAnalyzer(Analyzer):
    """Analyzer subclass.

    Attributes:
        n_samples (int): The amount of sampling.
        seed_sample (int): The seed of sampling random generator.
        seed_analyze (int): The seed of analyzer random generator.
        num_levels (int): The number of grid levels.

    """

    def __init__(self, n_samples=1000, seed_sample=7, seed_analyze=8,
                 num_levels=4):
        super().__init__()
        self.n_samples = n_samples
        self.seed_sample = seed_sample
        self.seed_analyze = seed_analyze
        self.num_levels = num_levels

    def analyze(self):
        """Initiate the analysis, and stores the result at data directory.

        Generates:
            Analysis result at 'acbm/data/output/morris.txt'.

        """

        X = m_sample(self.problem, self.n_samples, num_levels=self.num_levels,
                     seed=self.seed_sample)
        Y = ACBM.evaluate(X)
        si = m_analyze(self.problem, X, Y, num_levels=self.num_levels,
                       seed=self.seed_analyze)
        pickle.dump(si, open(self.path_output + 'morris.txt', 'wb'))
