#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 22:38:36 2024

@author: hessel
"""

import numpy as np
from ttmtdbdata import data as dt
from cnumba import cnjit

@cnjit(signature_or_function = 'f8(f8, f8, UniTuple(float64[:, :], 4))')
def TTmTDB_calc1(tt_jd, tt_jd2, data=dt):
    """
    Time difference between TT and TDB calculated from a series evaluation.
    
    The accuracy of the correction (cut off) is 10 us.

    Parameters
    ----------
    tt_jd : float
            Julian time in the TT (terrestrial time) timescale.

    Returns
    -------
    float
            The difference TT-TDB for the TT time, given in seconds.
    """
    result = 0
    T = ((tt_jd - 2451545.0) + tt_jd2) / 365250
    for power in range(4):
        n, C, w, phi = data[power]
        terms = C * pow(T, power) * np.sin(w*T + phi)
        result += np.sum(terms)
    return -1e-6*result

def TTmTDB_calc(tt1, tt2=0):
    return TTmTDB_calc1(tt1, tt2, dt)
