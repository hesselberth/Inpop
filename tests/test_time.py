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


filename_tdb = "../ephem/inpop21a_TDB_m100_p100_tt.dat"
filename_tcb = "../ephem/inpop21a_TCB_m100_p100_tcg.dat"
filename_not = "../ephem/inpop21a_TCB_m100_p100_bigendian.dat"


#  TDB

def test_file_open_failed_tdb():
    inpop = Inpop(filename_tdb, load=False)   
    inpop.file = None
    with pytest.raises(IOError) as excinfo:
        dt = inpop.TTmTDB(JD2000)

def test_jd_limits_tdb():
    inpop = Inpop(filename_tdb)
    with pytest.raises(ValueError) as excinfo:
        dt = inpop.TTmTDB(inpop.jd_beg - 1/86400)
    with pytest.raises(ValueError) as excinfo:
        dt = inpop.TTmTDB(inpop.jd_end + 1/86400)

def test_jd_inputs_tdb():
    inpop = Inpop(filename_tdb)
    ref = inpop.TTmTDB(JD2000 + 0.5)
    dt  = inpop.TTmTDB(np.array([JD2000 + 0.5]))
    assert((dt == ref).all())
    dt  = inpop.TTmTDB(np.array([JD2000, 0.5]))
    assert((dt == ref).all())
    with pytest.raises(ValueError) as excinfo:
        dt = inpop.TTmTDB(np.array([JD2000, 0.5, 0.0]))

def test_ts_tdb():
    inpop = Inpop(filename_tcb)
    with pytest.raises(KeyError) as excinfo:
        dt  = inpop.TTmTDB(JD2000)


# TCB

def test_file_open_failed_tcb():
    inpop = Inpop(filename_tcb, load=False)   
    inpop.file = None
    with pytest.raises(IOError) as excinfo:
        dt = inpop.TCGmTCB(JD2000)

def test_jd_limits_tcb():
    inpop = Inpop(filename_tcb)
    with pytest.raises(ValueError) as excinfo:
        dt = inpop.TCGmTCB(inpop.jd_beg - 1/86400)
    with pytest.raises(ValueError) as excinfo:
        dt = inpop.TCGmTCB(inpop.jd_end + 1/86400)

def test_jd_inputs_tcb():
    inpop = Inpop(filename_tcb)
    ref = inpop.TCGmTCB(JD2000 + 0.5)
    dt  = inpop.TCGmTCB(np.array([JD2000 + 0.5]))
    assert((dt == ref).all())
    dt  = inpop.TCGmTCB(np.array([JD2000, 0.5]))
    assert((dt == ref).all())
    with pytest.raises(ValueError) as excinfo:
        dt = inpop.TCGmTCB(np.array([JD2000, 0.5, 0.0]))

def test_ts_tcb():
    inpop = Inpop(filename_tdb)
    with pytest.raises(KeyError) as excinfo:
        dt  = inpop.TCGmTCB(JD2000)

def test_no_data_tcb():
    inpop = Inpop(filename_not)
    with pytest.raises(KeyError) as excinfo:
        dt  = inpop.TCGmTCB(JD2000)

