#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 21:07:07 2024

@author: hessel
"""

import os
import sys
sys.path.append(os.getcwd() + '/..')


from inpop.constants import AU, JD2000, DEG2RAD, Lg, T0, TDB0, Lb
from inpop import Inpop
import numpy as np
from os import path
import time
from .ttmtdb import TTmTDB_calc


def parse_testpo(filename):
    print(f"Loading {filename}")
    file = open(filename)
    lines = file.readlines()
    file.close()
    test = False
    JD = []
    T = []
    C = []
    X = []
    REF = []
    for line in lines:
        line=line.strip()
        if test:
            denum, date, jd, t, c, x, ref = line.split()
            denum = int(denum)
            if denum != 100:
                raise(ValueError("Bad DENUM value in testpo file."))
            JD.append(float(jd))
            T.append(int(t) - 1)
            C.append(int(c) - 1)
            X.append(int(x) - 1)
            REF.append(float(ref))
        if line.upper() == "EOT":
            test = True
    return np.array(JD), np.array(T), np.array(C), np.array(X), np.array(REF)
    

def pv_testpo(filename, mode, crosscheck=False):
    if mode:
        s = "memory"
    else:
        s = "disk"
    if crosscheck:
        c = "PV crosscheck"
    else:
        c = "PV check"
    print(f"Opening file in {s} mode for {c}")
    inpop = Inpop(filename, load=mode)
    print(inpop.path)

    # no open error
    n = len(inpop.constants)
    ts = inpop.timescale
    if ts == "TDB":
        ts_other = "TCB"
    elif ts == "TCB":
        ts_other = "TDB"
    else:
        raise(ValueError("Invalid timescale"))
    if crosscheck:
        tscheck = ts_other
    else:
        tscheck = ts
    
    testpo_dir = "."
    filename = path.basename(filename)
    parts = filename.split("_")
    testpo_filename = "testpo."+parts[0].upper()+"_"+tscheck
    testpo_path = path.join(testpo_dir, testpo_filename)
    if path.isfile(testpo_path):
        JD, T, C, X, REF = parse_testpo(testpo_path)
    else:
        raise(IOError(f"Failed to open testpo file {testpo_path}."))    

    n = len(JD)
    RES_PV = np.zeros(n, dtype=np.double)

    print(f"Computing {2*n} state vectors...")
    tstart = time.time()
    for i in range(n):
        pv = inpop.PV(JD[i], T[i], C[i], ts=tscheck)
        RES_PV[i] = pv.reshape(6)[X[i]]
    tstop = time.time()
    t_pv = tstop - tstart
    print(f"Elapsed time: {t_pv:.3f} s")
    PV_ERR = abs(RES_PV - REF)
    largest_PV = max(PV_ERR)
    print(f"Largest error: {largest_PV:.1e} AU")
    if largest_PV > 3e-13:
        raise(ValueError("Failed PV test."))
    
    print("Successfully passed the PV test (must be below 6e-13AU = 0.05m).")



def TTmTDB_theory(filename, mode):
    if mode:
        s = "memory"
    else:
        s = "disk"
    print(f"Opening file in {s} mode:")
    inpop = Inpop(filename, load=mode)
    assert(inpop.timescale == "TDB")

    # no open error
    testpo_dir = "."
    filename = path.basename(filename)
    parts = filename.split("_")
    testpo_filename = "testpo."+parts[0].upper()+"_TDB"
    testpo_path = path.join(testpo_dir, testpo_filename)
    assert(path.isfile(testpo_path))

    JD, T, C, X, REF = parse_testpo(testpo_path)

    n = len(JD)
    RES_T  = np.zeros(n, dtype=np.double)

    print(f"Computing {n} time conversions...")
    tstart = time.time()
    for i in range(n):
        RES_T[i] = inpop.TTmTDB(JD[i])
    tstop = time.time()
    t_t = tstop - tstart
    print(f"Elapsed time: {t_t:.3f} s")
    
    print(f"Estimating {n} time conversions based on GR approximation...")
    tstart = time.time()
    for i in range(n):
        REF[i] = TTmTDB_calc(JD[i])
    tstop = time.time()
    t_tc = tstop - tstart
    print(f"Elapsed time: {t_tc:.3f} s")
    
    T_ERR = abs(RES_T - REF)
    largest_T = max(T_ERR)
    print(f"Largest error: {largest_T:.10e} s (must be below 1 us).")
    if largest_T > 1e-6:
        raise(ValueError("Failed TTmTDB test"))
    
    print("Successfully passed the TTmTDB test.")
    print()
    

def TCGmTCB_calc(tt1, tt2=0):  # takes tt instead of tcg, good enough for test
    tcgmtt = TCGmTT = (Lg / (1 - Lg)) * ((tt1 - T0) + tt2) * 86400
    jd_tcg, jd_tcg2 = tt2, tt2 + tcgmtt / 86400
    ttmtdb = TTmTDB_calc(tt1, tt2)
    jd_tcb, jd_tcb2 = tt1, tt2 + ttmtdb / 86400
    tdbmtcb = -Lb * ((jd_tcb - T0) + jd_tcb2) * 86400 + TDB0  # jd_tcb
    return tcgmtt + ttmtdb + tdbmtcb


def TCGmTCB_theory(filename, mode):
    if mode:
        s = "memory"
    else:
        s = "disk"
    print(f"Opening file in {s} mode:")
    inpop = Inpop(filename, load=mode)
    assert(inpop.timescale == "TCB")

    # no open error
    testpo_dir = "."
    filename = path.basename(filename)
    parts = filename.split("_")
    testpo_filename = "testpo."+parts[0].upper()+"_TCB"
    testpo_path = path.join(testpo_dir, testpo_filename)
    assert(path.isfile(testpo_path))

    JD, T, C, X, REF = parse_testpo(testpo_path)

    n = len(JD)
    RES_T  = np.zeros(n, dtype=np.double)

    print(f"Computing {n} time conversions...")
    tstart = time.time()
    for i in range(n):
        RES_T[i] = inpop.TCGmTCB(JD[i])
    tstop = time.time()
    t_t = tstop - tstart
    print(f"Elapsed time: {t_t:.3f} s")
    
    print(f"Estimating {n} time conversions based on GR approximation...")
    tstart = time.time()
    for i in range(n):
        REF[i] = TCGmTCB_calc(JD[i])
    tstop = time.time()
    t_tc = tstop - tstart
    print(f"Elapsed time: {t_tc:.3f} s")
    
    T_ERR = abs(RES_T - REF)
    largest_T = max(T_ERR)
    print(f"Largest error: {largest_T:.10e} s (must be below 1 us).")
    if largest_T > 1e-6:
        raise(ValueError("Failed TTmTDB test"))
    
    print("Successfully passed the TTmTDB test.")
    print()
