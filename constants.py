#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 21:19:41 2023

@author: Marcel Hesselberth
"""

from numpy import pi

PI = pi
PI2 = 2 * pi

DEG2RAD = pi / 180
AM2RAD = pi / (180*60)
AS2RAD = pi / (180 * 60 * 60)
UAS2RAD = pi / (1000000 * 180 * 60 * 60)

SPD    = 86400                # seconds per day
CLIGHT = 2.99792458e8         # m/s
GS     = 1.32712440017987e20  # heliocentric gravitational constant, m3/s2
Lg     = 6.969290134e-10      #  TT -> TCG
T0     = 2443144.5003725      #  TT -> TCG
