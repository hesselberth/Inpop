#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Min Jan 06 21:37:00 2024

@author: Marcel Hesselberth
"""

# import os
# import sys
# sys.path.append(os.getcwd() + '/..')


from testlib import pv_testpo, TTmTDB_theory, TCGmTCB_theory


filename_tdb = "inpop21a_TDB_m100_p100_tt.dat"
filename_tcb = "inpop21a_TCB_m100_p100_tcg.dat"
filename_be  = "inpop21a_TCB_m100_p100_bigendian.dat"
filename_big = "inpop21a_TDB_m1000_p1000_littleendian.dat"

def test_all_modes_tdb():
    pv_testpo(filename_tdb, False)
    pv_testpo(filename_tdb, True)
    pv_testpo(filename_tdb, None)

def test_tcb():
    pv_testpo(filename_tcb, None)

def test_big_endian():
    pv_testpo(filename_be, None)
    
def test_big_file():
    pv_testpo(filename_big, None)

def test_crosscheck_tdb_tcb():
    pv_testpo(filename_tdb, False, True)

def test_crosscheck_tcb_tdb():
    pv_testpo(filename_tcb, None, True)

def test_TTmTDB():
    TTmTDB_theory(filename_tdb, None)

def test_TCGmTCB():
    TCGmTCB_theory(filename_tcb, None)

if __name__ == "__main__":
    test_all_modes_tdb()
    test_tcb()
    test_big_endian()
    test_big_file()
    test_crosscheck_tdb_tcb()
    test_crosscheck_tcb_tdb()
    test_TTmTDB()
    test_TCGmTCB()