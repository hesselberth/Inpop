#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 05 16:27:27 2025

@author: Marcel Hesselberth
"""

import os
import sys
sys.path.append(os.getcwd() + '/..')


from inpop import Inpop, lpath, config
import io, stat
import numpy as np
import pytest


filename       = "inpop21a_TDB_m100_p100_tt.dat"
filename_be    = "inpop21a_TCB_m100_p100_bigendian.dat"
filename_bad   = "../inpop21a_TCB.dat"
filename_worse = "foo.dat"
filename_uw    = "unwritable/" + filename


def test_config():
    ext      = config["inpopfile"]["ext"]
    default  = config["inpopfile"]["default"]
    base_url = config["ftp"]["base_url"]
    assert(ext=="dat")
    assert("inpop" in default)
    assert(base_url[:8]=="https://" and base_url[-1] == "/")
    
    
def test_bad_file():  # not found but contains path.sep
    with pytest.raises(FileNotFoundError) as excinfo:
        inpop = Inpop(filename_bad)


def test_worse_file():  # not found but no inpopXXy_
    with pytest.raises(IOError) as excinfo:
        inpop = Inpop(filename_worse)


def test_empty_file():  # found, good name but size 0 = not found
    ephem_path = "."
    emptyfilename = "inpop21a_EMPTY.dat"
    path = os.path.join(ephem_path, emptyfilename)
    file = open(path, "w")
    file.close()
    with pytest.raises(FileNotFoundError) as excinfo:
        inpop = Inpop(emptyfilename)
    os.unlink(path)


def test_writable_path():
        try:
            os.mkdir("writable")
        except:
            pass 
        test = os.path.join("writable", filename)
        inpop = Inpop(test)
        os.unlink(test)
        os.rmdir("writable")


def test_unwritable_path():
        try:
            os.mkdir("unwritable")
        except:
            pass
        permissions = os.stat("unwritable").st_mode
        os.chmod("unwritable", permissions & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
        test = os.path.join("unwritable", filename)    
        with pytest.raises(IOError) as excinfo:
            inpop = Inpop(test)
        try:
            os.unlink(test)
        except:
            pass
        else:
            raise (IOError("Created file in unwritable folder"))
        os.rmdir("unwritable")

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

print(np.get_printoptions())
np.set_printoptions(precision=20)
i = Inpop()
from inpop.constants import JD2000
print(i.LBR(JD2000))
print(repr(i))
