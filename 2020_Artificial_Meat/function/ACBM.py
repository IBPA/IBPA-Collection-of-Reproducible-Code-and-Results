# -*- coding: utf-8 -*-
"""ACBM.py description.

This script contains the ACBM model algorithm.

Author:
    Fangzhou Li - https://github.com/fangzhouli

"""

from math import log, ceil
import numpy as np


def acbm(conc_inoc, V_inoc, V_seed, conc_seed, V_b,
         concen_desired, m, f_ab, f_L, t_m, t_y, f_s, f_FM, V_c, rho_c, t_D,
         GCR_c, OUR_c, conc_aa2p, conc_nahco3, conc_sodium, conc_insulin,
         conc_trans, conc_fgf2, conc_tgfb, conc_glu, rho_m, f_O2, epsilon_bT,
         f_bT, T_e, T_d, W_Cv, epsilon_Hm, h, epsilon_BR, ACBM_Cv, T_b, T_c,
         epsilon_ACBMR, p_b, f_C, f_Sca, f_T, f_Q, f_B, f_O, D_r, I_D, L_e,
         I_EQ, C_b, C_basel, C_aa2p, C_nahco3, C_sodium, C_insulin, C_trans,
         C_fgf2, C_tgfb, C_O2, C_NG, C_NGP, C_L, C_PW, C_WF, C_BO):
    """This function takes model inputs and calculate ACBM cost

    Args:
        Each parameter is a float number

    Returns:
        (float): ACBM cost

    """

    def get_consumption(t_step, t_total, n_cell, r_cell):
        """This is a helper function to calculate consumption of a certain
        material.

        Args:
            t_step (float): Step size of each time interval.
            t_total (float): Total time of reaction.
            n_cell (int): The amount of cells.
            r_cell (float): The consumption rate for each cell.

        Returns:
            consumption (float): The total amount of consumption.

        """

        consumption = 0
        t = [i * t_step for i in range(ceil(t_total / t_step))] + [t_total]
        c_t = [2 ** (ti / t_step) * n_cell for ti in t]
        r_t = [r_cell * c for c in c_t]
        for i in range(1, len(c_t)):
            consumption += (t[i] - t[i - 1]) * r_t[i - 1]
        return consumption

    desired_n_cell_bioreactor = 1000 * V_b * concen_desired * 1000
    desired_n_cell_seed = 1000 * V_seed * conc_seed
    t_growth = log(desired_n_cell_bioreactor / desired_n_cell_seed, 2) * t_D
    b_BY = t_y / (t_growth + t_m)
    N_b = m / (desired_n_cell_bioreactor * V_c * rho_c * b_BY)
    glu_mat_consu = desired_n_cell_bioreactor * t_m * GCR_c
    glu_gro_consu = get_consumption(t_D, t_growth, desired_n_cell_seed,
                                    GCR_c)
    V_glu = (glu_mat_consu + glu_gro_consu) / conc_glu
    V_m = V_glu * N_b * b_BY

    # fixed manufaturing
    C_eq = N_b * f_ab * C_b * V_b ** f_s
    C_F = f_L * C_eq
    C_FM = f_FM * C_F

    # media
    C_mL = C_basel + C_aa2p * conc_aa2p + C_nahco3 * conc_nahco3 + \
        C_sodium * conc_sodium + C_insulin * conc_insulin + \
        C_trans * conc_trans + C_fgf2 * conc_fgf2 + C_tgfb * conc_tgfb
    C_mY = V_m * C_mL

    # oxygen
    O_2i = V_glu * rho_m * f_O2 / 0.032
    O_2g = get_consumption(t_D, t_growth, desired_n_cell_seed, OUR_c)
    O_2M = desired_n_cell_bioreactor * OUR_c * t_m
    O_2b = O_2i + O_2g + O_2M
    b_y = N_b * b_BY
    O_2 = O_2b * b_y
    C_O2Y = O_2 * 0.032 * C_O2 / 1000

    # energy
    C_EP = (0.09 * C_NG + 6.78) / 100
    C_bT = C_NGP / epsilon_bT / 100
    C_E = (1 - f_bT) * C_EP + f_bT * C_bT
    E_Hm = V_glu * N_b * b_BY * rho_m * (T_d - T_e) * W_Cv / epsilon_Hm
    E_BR = O_2 * h / epsilon_BR
    E_ACBMR = m * (T_b - T_c) * ACBM_Cv / epsilon_ACBMR

    # labor
    f_lab = f_C * f_Sca * f_T * f_Q * f_B * f_O
    P = p_b * N_b
    C_LAB = t_y * f_lab * C_L * P

    # water
    C_W = V_m * (C_PW + C_WF + C_BO) / 1000

    # finance
    C_D = C_F * D_r
    f_CRD = I_D * (1 + I_D) ** L_e / ((1 + I_D) ** L_e - 1)
    D_p = f_CRD * C_D
    f_CREQ = I_EQ * (1 + I_EQ) ** L_e / ((1 + I_EQ) ** L_e - 1)
    EQ_r = 1 - D_r
    C_TEQ = EQ_r * C_F
    EQ_p = f_CREQ * C_TEQ

    C_cap = D_p + EQ_p
    C_op = C_FM + C_mY + C_O2Y + C_E * (E_Hm + E_BR + E_ACBMR) + C_LAB + C_W
    return (C_op + C_cap) / m


def evaluate(param_values):
    Y = []
    for X in param_values:
        Y.append(acbm(*X))
    return np.array(Y)
