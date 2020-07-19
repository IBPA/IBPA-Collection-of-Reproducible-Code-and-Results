# -*- coding: utf-8 -*-
"""analyze.py description.

This script runs all the Analyzer modules defined in 'acbm/analyzers/'.

Author:
    Fangzhou Li - https://github.com/fangzhouli

"""


from analyzers import *


def analyze_all():
    """Call analyze methods of all analyzers.

    Generates:
        6 analysis results in 'acbm/data/output/'.

    """
    delta = DeltaAnalyzer()
    dgsm = DGSMAnalyzer()
    fast = FastAnalyzer()
    morris = MorrisAnalyzer()
    rbd_fast = RDBFastAnalyzer()
    sobol = SobolAnalyzer()

    delta.analyze()
    dgsm.analyze()
    fast.analyze()
    morris.analyze()
    rbd_fast.analyze()
    sobol.analyze()


if __name__ == '__main__':
    analyze_all()
