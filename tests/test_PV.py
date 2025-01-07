#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 05 22:21:07 2025

@author: Marcel Hesselberth
"""

from inpop import Inpop
from constants import JD2000
import numpy as np
import pytest


filename     = "inpop21a_TDB_m100_p100_tt.dat"
filename_be  = "inpop21a_TCB_m100_p100_bigendian.dat"
filename_bad = "foo.py"


def test_file_open_failed():
    inpop = Inpop(filename, load=False)   
    inpop.file = None
    with pytest.raises(IOError) as excinfo:
        pv = inpop.PV(JD2000, 12, 0)

def test_jd_limits():
    inpop = Inpop(filename)
    with pytest.raises(ValueError) as excinfo:
        pv = inpop.PV(inpop.jd_beg - 1/86400, 12, 0)
    with pytest.raises(ValueError) as excinfo:
        pv = inpop.PV(inpop.jd_end + 1/86400, 12, 0)

def test_jd_inputs():
    inpop = Inpop(filename)
    ref = inpop.PV(JD2000 + 0.5, "Saturn", "Earth")
    pv  = inpop.PV(np.array([JD2000 + 0.5]), "Saturn", "Earth")
    assert((pv == ref).all())
    pv  = inpop.PV(np.array([JD2000, 0.5]), "Saturn", "Earth")
    assert((pv == ref).all())
    with pytest.raises(ValueError) as excinfo:
        pv = inpop.PV(np.array([JD2000, 0.5, 0.0]), 12, 0)

def test_target_center_string():
    inpop = Inpop(filename)
    ref = inpop.PV(JD2000, 5, 2)
    pv  = inpop.PV(JD2000, "Saturn", "Earth")
    assert((pv == ref).all())
    with pytest.raises(KeyError) as excinfo:
        pv = inpop.PV(JD2000, SyntaxError, 2)
    with pytest.raises(KeyError) as excinfo:
        pv = inpop.PV(JD2000, "Alpha Centauri", 2)
    with pytest.raises(KeyError) as excinfo:
        pv = inpop.PV(JD2000, 5, "Polaris")

def test_ts_arg():
    inpop = Inpop(filename)
    ref = inpop.PV(JD2000 , "Saturn", "Earth")
    pv  = inpop.PV(JD2000, "Saturn", "Earth", ts="TDB")
    assert((pv == ref).all())
    ref = inpop.PV(JD2000 , "Saturn", "Earth")
    pv  = inpop.PV(JD2000, "Saturn", "Earth", ts="TCB")
    assert((pv != ref).any())
    assert(((pv - ref) < 1e-8).all())
    with pytest.raises(ValueError) as excinfo:
        pv  = inpop.PV(JD2000, "Saturn", "Earth", ts="TCG")


    
