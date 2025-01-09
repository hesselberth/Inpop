#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 21:19:41 2023

@author: Marcel Hesselberth
"""

from numpy import pi

mdays   = {1:31,2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31,
              11:30, 12:31}
wdays   = { 0:"Sun", 1:"Mon", 2:"Tue", 3:"Wed", 4:"Thu", 5:"Fri", 6:"Sat" }

PI = pi
PI2 = 2 * pi

DEG2RAD = pi / 180
AM2RAD  = pi / (180*60)
AS2RAD  = pi / (180 * 60 * 60)
UAS2RAD = pi / (1000000 * 180 * 60 * 60)

JD2000  = 2451545.0            # Start of the JD2000 epoch
MJD0    = 2400000.5            # For computing Modified Julian days

AU      = 149597870.7e3        # astronomical unit in m
SPD     = 86400                # seconds per day
CLIGHT  = 2.99792458e8         # m/s
GS      = 1.32712440017987e20  # heliocentric gravitational constant, m3/s2
Lb      = 1.550519768e-8       # TCB -> TDB
Kb      = 1 - Lb
LKb     = Lb / Kb
TDB0    = -6.55e-5             # seconds
TDB0_jd = TDB0 / SPD
Lg      = 6.969290134e-10      #  TT -> TCG
T0      = 2443144.5003725      #  definition of coordinate time scales
