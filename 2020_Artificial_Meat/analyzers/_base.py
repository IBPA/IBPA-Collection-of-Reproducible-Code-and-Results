# -*- coding: utf-8 -*-
"""_base.py description.

This module defines the base class for Analyzer that provides methods of
preparing the sensitivity analysis.

Author:
    Fangzhou Li - https://github.com/fangzhouli

"""

import os
import pandas as pd


class Analyzer:
    """Analyzer class, this class should only used by inheritance.

    Attributes:
        path_root (str): The absolute path to the root.
        path_data (str): The absolute path to the data.
        path_output (str): The absolute path to the output directory.
        problem (:obj:): A problem object used by SALib.

    """

    def __init__(self):
        self.path_root = os.path.abspath(os.path.dirname(__file__))
        self.path_data = self.path_root + '/../data/calculator.xlsx'
        self.path_output = self.path_root + '/../data/output/'
        self.problem = self.load_problem()

        if not os.path.exists(self.path_output):
            os.mkdir(self.path_output)

    def load_problem(self):
        """Load the problem attribute.

        Returns:
            (:obj:): A problem object.
        """

        # load calculator
        data = pd.read_excel(
            io=self.path_data,
            header=None)
        data.index = range(1, len(data) + 1)
        data.columns = [chr(i) for i in range(65, 88)]

        # load inputs
        conc_inoc = data['K'][4]
        V_inoc = data['K'][5]
        V_seed = data['K'][6]
        conc_seed = data['K'][7]
        V_b = data['K'][8]
        concen_desired = data['K'][10]
        m = data['K'][11]
        f_ab = data['K'][12]
        f_L = data['K'][13]
        t_m = data['K'][14]
        t_y = data['K'][15]
        f_s = data['K'][16]
        f_FM = data['K'][18]
        V_c = data['K'][24]
        rho_c = data['K'][25]
        t_D = data['K'][26]
        GCR_c = data['K'][27]
        OUR_c = data['K'][28]
        conc_aa2p = data['K'][32]
        conc_nahco3 = data['K'][33]
        conc_sodium = data['K'][34]
        conc_insulin = data['K'][35]
        conc_trans = data['K'][36]
        conc_fgf2 = data['K'][37]
        conc_tgfb = data['K'][38]
        conc_glu = data['K'][39]
        rho_m = data['K'][40]
        f_O2 = data['K'][41]
        epsilon_bT = data['K'][44]
        f_bT = data['K'][45]
        T_e = data['K'][51]
        T_d = data['K'][52]
        W_Cv = data['K'][53]
        epsilon_Hm = data['K'][54]
        h = data['K'][56]
        epsilon_BR = data['K'][57]
        ACBM_Cv = data['K'][59]
        T_b = data['K'][60]
        T_c = data['K'][61]
        epsilon_ACBMR = data['K'][62]
        p_b = data['K'][79]
        f_C = data['K'][83]
        f_Sca = data['K'][84]
        f_T = data['K'][85]
        f_Q = data['K'][86]
        f_B = data['K'][87]
        f_O = data['K'][88]
        D_r = data['K'][102]
        I_D = data['K'][103]
        L_e = data['K'][104]
        I_EQ = data['K'][105]
        C_b = data['N'][4]
        C_basel = data['N'][11]
        C_aa2p = data['N'][12]
        C_nahco3 = data['N'][13]
        C_sodium = data['N'][14]
        C_insulin = data['N'][15]
        C_trans = data['N'][16]
        C_fgf2 = data['N'][17]
        C_tgfb = data['N'][18]
        C_O2 = data['N'][25]
        C_NG = data['N'][28]
        C_NGP = data['N'][29]
        C_L = data['N'][32]
        C_PW = data['N'][36]
        C_WF = data['N'][37]
        C_BO = data['N'][38]

        # generate problem
        names = ['conc_inoc', 'V_inoc', 'V_seed', 'conc_seed', 'V_b',
                 'concen_desired', 'm', 'f_ab', 'f_L', 't_m', 't_y', 'f_s',
                 'f_FM', 'V_c', 'rho_c', 't_D', 'GCR_c', 'OUR_c', 'conc_aa2p',
                 'conc_nahco3', 'conc_sodium', 'conc_insulin', 'conc_trans',
                 'conc_fgf2', 'conc_tgfb', 'conc_glu', 'rho_m', 'f_O2',
                 'epsilon_bT', 'f_bT', 'T_e', 'T_d', 'W_Cv', 'epsilon_Hm', 'h',
                 'epsilon_BR', 'ACBM_Cv', 'T_b', 'T_c', 'epsilon_ACBMR', 'p_b',
                 'f_C', 'f_Sca', 'f_T', 'f_Q', 'f_B', 'f_O', 'D_r', 'I_D',
                 'L_e', 'I_EQ', 'C_b', 'C_basel', 'C_aa2p', 'C_nahco3',
                 'C_sodium', 'C_insulin', 'C_trans', 'C_fgf2', 'C_tgfb',
                 'C_O2', 'C_NG', 'C_NGP', 'C_L', 'C_PW', 'C_WF', 'C_BO']

        bounds = []
        for name in names:
            val = eval(name)
            bounds.append([val * 0.75, val * 1.25])

        return {
            'num_vars': len(names),  # 67
            'names': names,
            'bounds': bounds
        }
