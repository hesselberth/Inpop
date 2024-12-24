#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 21:07:07 2024

@author: hessel
"""


from inpop import Inpop
import numpy as np
from os import path
import sys, time


def parse_testpo(filename):
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
    

def test_TDB_file(filename, mode):
    if mode:
        s = "memory"
    else:
        s = "disk"
    print(f"Opening TDB file in {s} mode...")
    inpop = Inpop(filename, load=mode)
    # no open error
    print("File info:")
    print(inpop)
    n = len(inpop.constants)
    print(f"Inpop file contains {n} constants")
    if n<1:  # actually there should be several
        raise(ValueError("No constants in file"))
    
    testpo_dir = path.dirname(filename)
    filename = path.basename(filename)
    parts = filename.split("_")
    testpo_filename = "testpo."+parts[0].upper()+"_"+parts[1].upper()
    testpo_path = path.join(testpo_dir, testpo_filename)
    if path.isfile(testpo_path):
        JD, T, C, X, REF = parse_testpo(testpo_path)
    else:
        raise(IOError(f"Failed to open testpo file {testpo_path}."))

    n = len(JD)
    RES_PV = np.zeros(n, dtype=np.double)
    RES_T  = np.zeros(n, dtype=np.double)
    RES_Tc = np.zeros(n, dtype=np.double)

    print(f"Computing {2*n} state vectors...")
    tstart = time.time()
    for i in range(n):
        RES_PV[i] = inpop.PV(JD[i], T[i], C[i]).reshape(6)[X[i]]
    tstop = time.time()
    t_pv = tstop - tstart
    print(f"Elapsed time: {t_pv:.3f} s")
    PV_ERR = abs(RES_PV - REF)
    largest_PV = max(PV_ERR)
    print(f"Largest error: {largest_PV:.1e}")
    if largest_PV > 6e-12:
        raise(ValueError("Failed PV test."))
    
    print("Successfully passed the PV test (must be below 6e-12AU = 1 m).")

    if not inpop.has_time:
        print("Skipping TTmTDB test (data not present).")
        return

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
        RES_Tc[i] = Inpop.TTmTDB_calc(JD[i])
    tstop = time.time()
    t_tc = tstop - tstart
    print(f"Elapsed time: {t_tc:.3f} s")
    
    T_ERR = abs(RES_T - RES_Tc)
    largest_T = max(T_ERR)
    print(f"Largest error: {largest_T:.1e} s (must be below 10 us).")
    if largest_T > 1e-5:
        raise(ValueError("Failed PV test"))
    
    print("Successfully passed the TTmTDB test.")
    print()

def test_TDB_file_both_modes(filename):
    test_TDB_file(filename, False)
    test_TDB_file(filename, True)


print("Running Inpop test...\n")
filename = "inpop21a_TDB_m100_p100_tt.dat"
test_TDB_file_both_modes(filename)



