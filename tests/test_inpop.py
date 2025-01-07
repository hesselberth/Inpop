#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 05 16:27:27 2025

@author: Marcel Hesselberth
"""

import os
import sys
sys.path.append(os.getcwd() + '/..')

from inpop import Inpop
import io
import numpy as np
import pytest


filename     = "../ephem/inpop21a_TDB_m100_p100_tt.dat"
filename_be  = "../ephem/inpop21a_TCB_m100_p100_bigendian.dat"
filename_bad = "../ephem/ttmtdbdata.py"


def test_bad_file():
    with pytest.raises(IOError) as excinfo:
        inpop = Inpop(filename_bad)


def test_other_endianness():
    inpop = Inpop(filename_be)
    assert("KSIZER" in inpop.constants)
    assert(abs(inpop.EMRAT - 81.30056789872074) < 1e-14)
    assert(inpop.DENUM == 100)
    inpop.close()


def test_constructor_file_mode():
    inpop = Inpop(filename, load = False)
    assert(inpop.mem == False)
    assert(isinstance(inpop.file, (io.RawIOBase, io.BufferedIOBase)))
    assert(inpop.timescale == "TDB")
    assert(inpop.DENUM == 100)
    inpop.close()


def test_constructor_memory_mode():
    inpop = Inpop(filename, load = True)
    assert(inpop.mem == True)
    assert(isinstance(inpop.data, np.ndarray))
    assert(inpop.timescale == "TDB")
    assert(inpop.DENUM == 100)
    inpop.close()


def test_constructor_auto_mode():
    inpop = Inpop(filename)
    assert(inpop.mem == True)
    assert(isinstance(inpop.data, np.ndarray))
    assert(inpop.timescale == "TDB")
    assert(inpop.DENUM == 100)
    inpop.close()

def test_info_str():
    inpop = Inpop(filename)
    s = str(inpop)
    inpop.close()
    assert("TDB" in s and "endian" in s and "100" in s)

def test_constants():
    inpop = Inpop(filename)
    n = len(inpop.constants)
    assert(n > 0)
    inpop.close()
    
    