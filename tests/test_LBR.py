#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 05 22:21:07 2025

@author: Marcel Hesselberth
"""

import os
import sys
sys.path.append(os.getcwd() + '/..')


from inpop import Inpop
from constants import JD2000
import numpy as np
import pytest


filename = "inpop21a_TDB_m100_p100_tt.dat"


def test_file_open_failed():
    inpop = Inpop(filename, load=False)   
    inpop.file = None
    with pytest.raises(IOError) as excinfo:
        lbr = inpop.LBR(JD2000)

def test_jd_limits():
    inpop = Inpop(filename)
    with pytest.raises(ValueError) as excinfo:
        lbr = inpop.LBR(inpop.jd_beg - 1/86400)
    with pytest.raises(ValueError) as excinfo:
        lbr = inpop.LBR(inpop.jd_end + 1/86400)

def test_jd_inputs():
    inpop = Inpop(filename)
    ref = inpop.LBR(JD2000 + 0.5)
    lbr  = inpop.LBR(np.array([JD2000 + 0.5]))
    assert((lbr == ref).all())
    lbr  = inpop.LBR(np.array([JD2000, 0.5]))
    assert((lbr == ref).all())
    with pytest.raises(ValueError) as excinfo:
        lbr = inpop.LBR(np.array([JD2000, 0.5, 0.0]))

def test_ts_arg():
    inpop = Inpop(filename)
    ref = inpop.LBR(JD2000, rate = False)
    lbr  = inpop.LBR(JD2000, ts="TDB", rate = False)
    assert((lbr == ref).all())
    ref = inpop.LBR(JD2000, rate = False)
    lbr  = inpop.LBR(JD2000, ts="TCB", rate = False)
    assert((lbr != ref).any())
    assert(((lbr - ref) < 2e-8).all())
    with pytest.raises(ValueError) as excinfo:
        lbr  = inpop.LBR(JD2000, ts="TCG")


    
