#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 21:07:07 2024

@author: hessel
"""

from constants import AU, JD2000, DEG2RAD
from inpop import Inpop
import numpy as np
from os import path
import time
import urllib.request

from dtmath import RJD
#from dt import TDBTime, JD2000
from ttmtdb import TTmTDB_calc

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
    

def inpop_test(filename, mode, crosscheck=False):
    if mode:
        s = "memory"
    else:
        s = "disk"
    if crosscheck:
        c = "PV crosscheck"
    else:
        c = "PV check"
    print(f"Opening file in {s} mode for {c}:")
    inpop = Inpop(filename, load=mode)
    # no open error
    print("File info:")
    print(inpop)
    n = len(inpop.constants)
    print(f"Inpop file contains {n} constants")
    if n<1:  # actually there should be several
        raise(ValueError("No constants in file"))
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
    
    print("Testing lower bound exception on JD... ", end="")
    try:
        inpop.PV(np.array([inpop.jd_beg, -1/86400]), "emb", "ssb")
    except ValueError:
        print("OK")
    else:
        raise(AssertionError("Failed."))
    print("Testing upper bound exception on JD... ", end="")
    try:
        inpop.PV(np.array([inpop.jd_end, 1/86400]), "emb", "ssb")
    except ValueError:
        print("OK")
    else:
        raise(AssertionError("Failed."))
    testpo_dir = path.dirname(filename)
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
    RES_T  = np.zeros(n, dtype=np.double)
    RES_Tc = np.zeros(n, dtype=np.double)

    print(f"Computing {2*n} state vectors...")
    tstart = time.time()
    e=0
    for i in range(n):
        pv = inpop.PV(JD[i], T[i], C[i], ts=tscheck)
        RES_PV[i] = pv.reshape(6)[X[i]]
        if X[i]<3 and ((T[i] == 2 and C[i] == 9) or (T[i] == 9 and  C[i]== 2)):
            err = RES_PV[i] - REF[i]
        #     err = err / np.linalg.norm(pv[0])
            err = abs(err*AU)
            if err > .00004:
                print(T[i], C[i], X[i], RJD(JD[i]), RES_PV[i]*AU, REF[i]*AU)
                e+=1
    tstop = time.time()
    t_pv = tstop - tstart
    print(f"Elapsed time: {t_pv:.3f} s")
    PV_ERR = abs(RES_PV - REF)
    largest_PV = max(PV_ERR)
    print(f"Largest error: {largest_PV:.1e} AU")
    if largest_PV > 1e-12:
        raise(ValueError("Failed PV test."))
    
    print("Successfully passed the PV test (must be below 6e-13AU = 0.1m).")

    if not inpop.has_time:
        print("Skipping TTmTDB test (data not present).")
        return
    if inpop.timescale == "TCB":
        return  # no ref
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
        RES_Tc[i] = TTmTDB_calc(JD[i])
    tstop = time.time()
    t_tc = tstop - tstart
    print(f"Elapsed time: {t_tc:.3f} s")
    
    T_ERR = abs(RES_T - RES_Tc)
    largest_T = max(T_ERR)
    print(f"Largest error: {largest_T:.1e} s (must be below 1 us).")
    if largest_T > 1e-6:
        raise(ValueError("Failed TTmTDB test"))
    
    print("Successfully passed the TTmTDB test.")
    print()




def require(url, filepath=""):
    if filepath == "":
        filepath = url.rsplit("/", 1)[-1]
    if path.exists(filepath):
        return
    urllib.request.urlretrieve(url, filepath)
    


def test_all():
    print("Running Inpop test...\n")
    print("Little Endian")
    require("https://ftp.imcce.fr/pub/ephem/planets/inpop21a/inpop21a_TDB_m100_p100_tt.dat")
    require("https://ftp.imcce.fr/pub/ephem/planets/inpop21a/testpo.INPOP21A_TCB")
    require("https://ftp.imcce.fr/pub/ephem/planets/inpop21a/testpo.INPOP21A_TDB")

    filename = "inpop21a_TDB_m100_p100_tt.dat"
    inpop_test(filename, False, False)
    inpop_test(filename, True, False)
    inpop_test(filename, False, True)
    inpop_test(filename, True, True)

if __name__ == "__main__":
    test_all()

#inpop_tcb = Inpop("inpop21a_TCB_m100_p100_tcg.dat")
#test_file_all_modes("inpop21a_TDB_m100_p100_tt.dat")
# print()
# print("Big Endian")
# test_file_all_modes("inpop21a_TCB_m100_p100_bigendian.dat")
# print()
# print("TCG")
# test_file_all_modes("inpop21a_TCB_m100_p100_tcg.dat")
# print()
# print("Large file")
# test_file_all_modes("inpop21a_TDB_m1000_p1000_littleendian.dat")
# print()
# # print("Successfully passed all tests.")
inpop = Inpop("inpop21a_TDB_m100_p100_tt.dat")
print(inpop.LBR(JD2000)/DEG2RAD)
inpop.close()