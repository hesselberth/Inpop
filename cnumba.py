#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:27:07 2024

@author: Marcel Hesselberth
"""


# =============================================================================
# Conditional Numba.
# If Numba is available, cjit is just jit and cnjit is njit.
# If Numba is not available, these decorators return the original function
# (do nothing).
# =============================================================================


try:
    from numba import jit, njit
except:
    numba_acc = False
else:
    numba_acc = True


def cjit(*cjit_args, **cjit_kwargs):
    def jit_func(f):
        if numba_acc:
            return jit(*cjit_args, **cjit_kwargs)(f)
        else:
            return f
    return jit_func

def cnjit(*cjit_args, **cjit_kwargs):
    def jit_func(f):
        if numba_acc:
            return njit(*cjit_args, **cjit_kwargs)(f)
        else:
            return f
    return jit_func

def timer(f):
    from time import time
    def timedf(*args, **kwargs):
        tstart = time()
        y = f(*args, **kwargs)
        tstop = time()
        print(f"timer {tstop-tstart} s\n")
        return y
    return timedf
