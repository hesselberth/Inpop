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
from inpop.constants import JD2000
import numpy as np
import pytest


filename_tdb = "inpop21a_TDB_m100_p100_tt.dat"
filename_tcb = "inpop21a_TCB_m100_p100_tcg.dat"
filename_not = "inpop21a_TCB_m100_p100_bigendian.dat"


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
    with pytest.raises(TypeError) as excinfo:
        pv = inpop.TTmTDB([])  # wrong type
    with pytest.raises(TypeError) as excinfo:
        pv = inpop.TTmTDB(())  # wrong length
    with pytest.raises(TypeError) as excinfo:
        pv = inpop.TTmTDB(np.array([]))  # wrong length


def test_ts_tdb():
    inpop = Inpop(filename_tcb)
    with pytest.raises(KeyError) as excinfo:
        dt = inpop.TTmTDB(JD2000)


def test_rate_tdb():
    inpop = Inpop(filename_tdb)
    ref = inpop.TTmTDB(JD2000)
    assert(ref.shape == ())  # scalar
    ref = inpop.TTmTDB(JD2000, rate = False)
    assert(ref.shape == ())
    ref = inpop.TTmTDB(JD2000, rate = True)
    assert(ref.shape == (2,))
    inpop.close()    


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
    with pytest.raises(TypeError) as excinfo:
        pv = inpop.TCGmTCB([])  # wrong type
    with pytest.raises(TypeError) as excinfo:
        pv = inpop.TCGmTCB(())  # wrong length
    with pytest.raises(TypeError) as excinfo:
        pv = inpop.TCGmTCB(np.array([]))  # wrong length


def test_ts_tcb():
    inpop = Inpop(filename_tdb)
    with pytest.raises(KeyError) as excinfo:
        dt  = inpop.TCGmTCB(JD2000)


def test_no_data_tcb():
    inpop = Inpop(filename_not)
    with pytest.raises(KeyError) as excinfo:
        dt = inpop.TCGmTCB(JD2000)


def test_rate_tcb():
    inpop = Inpop(filename_tcb)
    ref = inpop.TCGmTCB(JD2000)
    assert(ref.shape == ())  # scalar
    ref = inpop.TCGmTCB(JD2000, rate = False)
    assert(ref.shape == ())
    ref = inpop.TCGmTCB(JD2000, rate = True)
    assert(ref.shape == (2,))
    inpop.close()    
