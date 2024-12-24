#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compact library for reading INPOP files.

Created on Fri Dec  4 14:16:35 2024

@author: Marcel Hesselberth

Version: 0.2
"""

from constants import Lg, T0
from cnumba import cnjit
import numpy as np
import struct
from os import SEEK_END
from sys import byteorder, exit


@cnjit(signature_or_function = 'UniTuple(float64[:], 2)(float64, int64)')
def chpoly(x, degree):
    """
    Evaluate the Chebyshev polynomial and its derivatives at x.
    
    Best algorithm according to https://arxiv.org/abs/1312.5677v2 .
    
    Parameters
    ----------
    x      : float.
             Domain [-1, 1].
    degree : int.
             Degree of the polynomial.

    Returns
    -------
    Polynomials T and (derivative) D.
    Both are arrays of length <degree>.
    """
    T = np.ones(degree, dtype=float)
    D = np.zeros(degree, dtype=float)
    T[1] = x
    D[1] = 1
    for i in range(2, degree):
        T[i] = 2.0 * x * T[i-1] - T[i-2]
        D[i] = 2.0 * T[i-1] + 2.0 * x * D[i-1] - D[i-2]
        # alternative w. 1 term less but a division by 1-t**2
        # D[i] = (-i*t*T[i] + i*T[i-1]) / (1-t*t)
    return T, D


@cnjit(signature_or_function = 'f8[:,:](f8, i4, i4, i4, f8[:], f8, f8, i4, i4)')
def calcm(jd, offset, ncoeffs, ngranules, data, \
          jd_beg, interval, nrecords, recordsize):
    """
    Compute a state vector (3-vector and its derivative) from  data in memory.

    This is the INPOP decoding routine common to the calculations, whether
    6d (position-velocity), 3d (libration angles) or 1d (time).
    calcm is an accelerated version of Inpop.calc1.

    Parameters
    ----------
    jd         : float
                 Julian date
    offset     : int
                 coef_ptr[0]
    ncoeffs    : int
                 coef_ptr[1]
    ngranules  : int
                 coef_ptr[2]
    data       : ndarray
                 self.data
    jd_beg     : float
                 self.jd_beg
    interval   : float
                 self.interval
    nrecords   : int
                 self.nrecords
    recordsize : int
                 self.recordsize

    Returns
    -------
    ndarray
        2x3 matrix pos, vel

    """
    record = int((jd - jd_beg)//interval) + 1
    if record < nrecords: record += 1
    raddr = record*recordsize
    jdl = data[raddr]
    span = interval / ngranules
    granule = int((jd - jdl) // span)
    jd0 = jdl + granule * span
    tc = 2 * ((jd-jd0) / span) - 1
    gaddr = int(raddr+(offset-1+3*granule*ncoeffs))
    cx = data[gaddr               : gaddr +     ncoeffs]
    cy = data[gaddr +     ncoeffs : gaddr + 2 * ncoeffs]
    cz = data[gaddr + 2 * ncoeffs : gaddr + 3 * ncoeffs]
    T, D = chpoly(tc, ncoeffs)
    px = np.dot(cx, T)
    py = np.dot(cy, T)
    pz = np.dot(cz, T)
    vx = np.dot(cx, D) * ngranules
    vy = np.dot(cy, D) * ngranules
    vz = np.dot(cz, D) * ngranules
    return np.array([[px, py, pz], [vx, vy, vz]], dtype=np.double)


class Inpop:
    """Decode Inpop .dat files and compute planetary positions."""
    
    bodycodes = {"mercury":0, "venus":1, "earth":2, "mars":3, "jupiter":4,
                 "saturn":5, "uranus":6, "neptune":7, "pluto":8, "moon":9,
                 "sun":10, "ssb":11, "emb":12}
    
    def __init__(self, filename, load=True):
        """
        Inpop constructor.
        
        Class to compute state vectors (planetary position and velocity) from
        the 4d INPOP ephemeris. Data is read from the .dat file using the INPOP
        file format and may have little or big endian byte order.

        Parameters
        ----------
        path : string
               Path of an INPOP .dat file
        load : bool, optional
               If True, the file is completely loaded to memory.
               If false, the file is accessed fully through seek operations,
               The default is True.

        Returns
        -------
        None.

        """
        self.path = filename
        self.file = None
        if byteorder == "little":
            self.machine_byteorder = "<"
            self.opposite_byteorder = ">"
        else:
            self.machine_byteorder = ">"
            self.opposite_byteorder = "<"
        self.byteorder = self.machine_byteorder
        self.mem = load
        self.open()


    def open(self):
        """
        Open the binary INPOP file.
        
        Read the header information, the constant values and initialize the
        lookup of Chebyshev polynomials. Some important variables are always
        present (AU, EMRAT, DENUM) and become member variables. According to
        specification 2.0 INPOP files can also contain asteroid information.
        Such files are not in the public domain, hence retrieving asteroid
        information is not (yet) implemented.

        Returns
        -------
        None.

        """
        self.file = open(self.path, 'rb')  # INPOP files are binary
        
        # Decode the header record
        header_spec   = f"{self.byteorder}252s2400sdddidd36ii3ii3i"
        header_struct = struct.Struct(header_spec)
        bytestr       = self.file.read(header_struct.size)
        hb            = header_struct.unpack(bytestr)  # header block
        self.DENUM    = hb[44]  # must be 100 for INPOP
        if self.DENUM != 100:
            self.file.seek(0)
            self.byteorder = self.opposite_byteorder
            header_spec    = f"{self.byteorder}252s2400sdddidd36ii3ii3i"
            header_struct  = struct.Struct(header_spec)
            bytestr        = self.file.read(header_struct.size)
            hb             = header_struct.unpack(bytestr)  # header block
            self.DENUM     = hb[44]
            if self.DENUM  != 100:
                raise(IOError("Can't determine INPOP file byteorder."))

        self.jd_struct  = struct.Struct(f"{self.byteorder}dd") # julian dates

        self.label      = []  # ephemeris label
        self.label.append(hb[0][:84].decode().strip())
        self.label.append(hb[0][84:168].decode().strip())
        self.label.append(hb[0][168:].decode().strip())

        const_names     = [hb[1][6*i:6*(i+1)] for i in range(400)]

        self.jd_beg     = hb[2]      # julian start date
        self.jd_end     = hb[3]      # julian end date
        self.interval   = hb[4]      # julian interval
        self.num_const  = hb[5]      # number of constants in the second record
        self.AU         = hb[6]      # Astronomical unit
        self.EMRAT      = hb[7]      # Mearth / Mmoon
        self.coeff_ptr  = [(hb[8+3*i:8+3*i+3]) for i in range(12)]
        self.DENUM      = hb[44]     # ephemeris ID
        self.librat_ptr = hb[45:48]  # libration pointer
        self.recordsize = hb[48]     # size of the record in bytes
        self.TTmTDB_ptr = hb[49:52]  # time transformation TTmTDB or TCGmTCB

        # these are the location, number of coefficients and number of granules
        # for the 12 bodies.
        self.coeff_ptr  = np.array(self.coeff_ptr, dtype = int)
        
        # these are the location, number of coefficients and number of granules
        # for the libration angles of the moon.
        self.librat_ptr  = np.array(self.librat_ptr, dtype = int)

        # these are the location, number of coefficients and number of granules
        # for the mapping of  TT-TDB or TCG-TCB
        self.TTmTDB_ptr  = np.array(self.TTmTDB_ptr, dtype = int)
        
        # Decode the constant record
        self.file.seek(self.recordsize*8)
        const_struct   = struct.Struct(f"{self.byteorder}%id"%(self.num_const))
        bytestr        = self.file.read(const_struct.size)
        cb             = const_struct.unpack(bytestr)
        const_names    = const_names[:self.num_const]
        self.constants = {const_names[i].strip().decode():cb[i] \
                          for i in range(self.num_const)}
        
        self.version  = self.constants["VERSIO"]  # ephemerus version number
        self.fversion = self.constants["FVERSI"]  # file version number (0)
        self.format   = self.constants["FORMAT"]  # details about file contents
        self.ksizer   = int(self.constants["KSIZER"])  # numbers per record

        # Decode file format
        self.has_vel       = (self.format//1%10)   == 0
        self.has_time      = (self.format//10)%10  == 1
        self.has_asteroids = (self.format//100)%10 == 1

        # Use the single unit base and transform where necessary
        self.unit_time  = "s"
        self.unit_angle = "rad"
        self.unit_pos   = "au"
        self.unit_vel   = "au/day"

        if self.constants["UNITE"] == 0:
            self.unite = 0
            self.unit_pos_factor = 1.0 
            self.unit_vel_factor = 2.0 / (self.interval)
        else:
            self.unite = 1
            self.unit_pos_factor = 1.0 / self.AU
            self.unit_vel_factor = 2.0 / (self.AU * self.interval)

        # If no timescale is found it is TDB (file version 1.0)
        if "TIMESC" in self.constants:
            if self.constants["TIMESC"] == 0:
                self.timescale = "TDB"
            else:
                self.timescale = "TCB"
        else:
            self.timescale = "TDB"

        self.nrecords = int((self.jd_end - self.jd_beg) / self.interval)
        
        if self.mem:
            self.load()
            self.file.close()
            self.file = None

        self.earthfactor = -1 / (1 + self.EMRAT)
        self.moonfactor  = self.EMRAT / (1 + self.EMRAT)


    def load(self):
        """
        Load the INPOP file in memory.
        
        This option speeds up the calculations by avoiding file operations.
        This also allows Numba acceleration.

        Returns
        -------
        None.

        """
        self.file.seek(0, SEEK_END)
        size = self.file.tell()
        self.file.seek(0)
        if size % 8 != 0:
            raise(ValueError("INPOP File has wrong length."))
        data = np.frombuffer(self.file.read(size), dtype=np.double)
        if self.byteorder != self.machine_byteorder:
            data = data.byteswap()  # Changes data (newbyteorder changes view)
        self.data = np.copy(data)   # Changes array status for Numba


    def info(self):
        """
        Generate a string containing information about the INPOP file.

        Returns
        -------
        s : string
            
        """
        if self.byteorder == '>':
            b = "Big-endian"
        else:
            b = "Little-endian"
        s  = f"INPOP file             {self.path}\n"
        s += f"Byte order             {b}\n"
        s += f"Label                  {self.label}\n"
        s += f"JDbeg, JDend, interval {self.jd_beg}, {self.jd_end}, "
        s += f"{self.interval}\n"
        s += f"record_size            {self.recordsize}\n"
        s += f"num_const              {self.num_const}\n"
        s += f"AU, EMRAT              {self.AU}, {self.EMRAT}\n"
        s += f"DENUM                  {self.DENUM}\n"
        s += f"librat_ptr             {self.librat_ptr}\n"
        s += f"TTmTDB_ptr             {self.TTmTDB_ptr}\n"
        s += f"version                {self.version}\n"
        s += f"fversion               {self.fversion}\n"
        s += f"format                 {self.format}\n"
        s += f"KSIZER                 {self.ksizer}\n"
        s += f"UNITE                  {self.unite}\n"

        s += f"has_vel                {self.has_vel}\n"
        s += f"has_time               {self.has_time}\n"
        s += f"has_asteroids          {self.has_asteroids}\n"
        
        s += f"unit_pos               {self.unit_pos}\n"
        s += f"unit_vel               {self.unit_vel}\n"
        s += f"unit_time              {self.unit_time}\n"
        s += f"unit_angle             {self.unit_angle}\n"
        s += f"timescale              {self.timescale}"
        #s += f"\ncoeff_ptr:\n{self.coeff_ptr}"
        return s


    def __str__(self):
        """
        Enable printing of an Inpop instance.

        Returns
        -------
        string info()
        """
        return self.info()


    def calc1(self, jd, coeff_ptr):
        """
        Calculate a state vector for a single body.

        This is the Inpop decoding routine common to the calculations, whether
        6d (position-velocity), 3d (libration angles) or 1d (time).
        The file record is located and checked and subsequently the INPOP
        granule with the coefficients is seeked. Based on the granule size
        the Chebyshev time tc is calculated. If the file is loaded in memory,
        numba-accelerated calcm is called.

        Parameters
        ----------
        jd        : float
                    Date in ephemeris time.
        offset    : int
                    Record offset in the file.
        ncoeffs   : int
                    Number of Chebyshev coefficients
        ngranules : int
                    Number of granules

        Returns
        -------
        2x3 matrix pos, vel
        """
        if jd < self.jd_beg or jd > self.jd_end:
            raise(ValueError("Julian date must be between %.1f and %.1f." \
                             % (self.jd_beg, self.jd_end)))
        offset, ncoeffs, ngranules = coeff_ptr
        if self.mem:
            return calcm(jd, offset, ncoeffs, ngranules, \
                         self.data, self.jd_beg, self.interval, \
                         self.nrecords, self.recordsize)
        else:  # file based
            if not self.file:
                raise(IOError(f"Ephemeris file ({self.path}) not open."))
        record = int((jd - self.jd_beg)//self.interval) + 1
        if record < self.nrecords: record += 1
        raddr = record*self.recordsize*8  # locate record
        self.file.seek(raddr)
        bytestr = self.file.read(self.jd_struct.size)  # read record limits
        jdl, jdh = self.jd_struct.unpack(bytestr)
        assert(jd>=jdl and jd<=jdh)  # check
        span = self.interval / ngranules
        granule = int((jd - jdl) // span)  # compute the granule
        jd0 = jdl + granule * span
        tc = 2 * ((jd-jd0) / span) - 1  # Chebyshev argument for the granule
        assert(tc>=-1 and tc <=1)
        gaddr = int(raddr+(offset-1 + 3*granule*ncoeffs)*8)  # -1 for C arrays
        self.file.seek(gaddr)  # read 3 * ncoeffs 8 bit doubles
        coeffs = np.frombuffer(self.file.read(24*ncoeffs), dtype=np.double)
        coeffs = coeffs.newbyteorder(self.byteorder)
        coeffs.resize((3, ncoeffs))  # 3 x ncoeffs matrix
        T, D = chpoly(tc, ncoeffs)  # 2 x ncoeffs
        pos = np.dot(coeffs, T)
        vel = np.dot(coeffs, D) * ngranules
        return np.array([pos, vel], dtype = np.double)


    def PV1(self, jd, body):
        """
        Compute the state of a single body in the ICRF frame.
        
        The state consists of 2 3-vectors: position and velocity (relative to
        the solar system barycenter). The position is given in AU, the
        velocity in AU/day.

        Parameters
        ----------
        jd : np.double (or float)
             Julian date in ephemeris time. INPOP is distributed in TDB and TCB.
             timescales (see self.timescale).
        body : integer between 0 and 12
        
        0:  Mercury
        1:  Venus 
        2:  Earth
        3:  Mars
        4:  Jupiter
        5:  Saturn
        6:  Uranus
        7:  Neptune
        8:  Pluto 
        9:  Moon
        10: Sun
        11: SSB
        12: EMB

        Returns
        -------
        2x3 matrix [P, V].
        Error upon failure (no ephemeris file found, time outside ephemeris,
        body code invalid.

        """
        if body < 0 or body > 12:
            raise(LookupError("Body code must be between 0 and 12."))
        if body == 11:
            return np.zeros(6).reshape((2, 3))
        if body == 12:
            body = 2
        pv = self.calc1(jd, self.coeff_ptr[body])
        pv[0] *= self.unit_pos_factor
        pv[1] *= self.unit_vel_factor
        return pv


    def PV(self, jd, t, c= 11):
        """
        Position and velocity of a target t relative to center c in the ICRF.

        Positions and velocities are computed using the Chebyshev polynomials
        and their derivatives. The position is given in AU, the velocity
        in AU/day.
        
        Parameters
        ----------
        jd : np.double (or float)
             Julian date in ephemeris time. INPOP is distributed in TDB and TCB.
             timescales (see self.timescale).
        t, c : integer between 0 and 12
        Target body and the Center from which it is observed.
        
        0:  Mercury
        1:  Venus 
        2:  Earth
        3:  Mars
        4:  Jupiter
        5:  Saturn
        6:  Uranus
        7:  Neptune
        8:  Pluto 
        9:  Moon
        10: Sun
        11: SSB
        12: EMB

        Returns
        -------
        2x3 matrix [P, V].
        Error upon failure (no ephemeris file found, time outside ephemeris,
        body code invalid.
        """
        if not isinstance(t, (int, np.integer)):
            try:
                t = Inpop.bodycodes[t.lower()]
            except:
                print(f"Unknown target ({t}).\nValid targets are:")
                inverse = {Inpop.bodycodes[x]:x for x in Inpop.bodycodes.keys()}
                for i in range(13):
                    print(f"{i:2d} {inverse[i]}")
                raise(KeyError) from None  # pep-0409
        if not isinstance(c, (int, np.integer)):
            try:
                c = Inpop.bodycodes[c.lower()]
            except:
                print(f"Unknown center ({c}).\nValid centers are:")
                inverse = {Inpop.bodycodes[x]:x for x in Inpop.bodycodes.keys()}
                for i in range(13):
                    print(f"{i:2d} {inverse[i]}")
                raise(KeyError) from None  # pep-0409        
        if t == c:
            return np.zeros(6).reshape((2, 3))
        if t == 2:
            target = self.PV1(jd, 9) * self.earthfactor
            if c == 9:
                center = self.PV1(jd, 9) * self.moonfactor
            else:
                target += self.PV1(jd, 2)
                center = self.PV1(jd, c)
        elif t == 9:
            target = self.PV1(jd, 9) * self.moonfactor
            if c == 2:
                center = self.PV1(jd, 9) * self.earthfactor
            else:
                target += self.PV1(jd, 2)
                center = self.PV1(jd, c)
        else:
            target = self.PV1(jd, t)
            if c == 2:
                center = self.PV1(jd, 9) * self.earthfactor \
                    + self.PV1(jd, 2)
            elif c == 9:
                center = self.PV1(jd, 9) * self.moonfactor \
                    + self.PV1(jd, 2)
            else:
                center = self.PV1(jd, c)
        return target - center


    def LBR(self, jd):
        """
        Physical libration angles of the moon.

        Parameters
        ----------
        jd : np.double (or float)
             Julian time in ephemeris time. INPOP is distributed in TDB and TCB
             timescales (see self.timescale).

        Returns
        -------
        np.array(3, dype="float")
             The 3 physical libration angles in radians
        """
        return self.calc1(jd, self.librat_ptr)[0]


    @cnjit(signature_or_function='float64(float64)')
    def TTmTDB_calc(tt_jd):  # truncated at <10 us presicion
        """
        Time difference between TT and TDB calculated from a series evaluation.
        
        The accuracy of the correction (cut off) is 10 us.

        Parameters
        ----------
        tt_jd : float
                Julian time in the TT (terrestrial time) timescale.

        Returns
        -------
        float
                The difference TT-TDB for the TT time, given in seconds.
        """
        T = (tt_jd - 2451545.0) / 36525
        ttmtdb = -0.001657 * np.sin (628.3076 * T + 6.2401)  \
                - 0.000022 * np.sin (575.3385 * T + 4.2970)  \
                - 0.000014 * np.sin (1256.6152 * T + 6.1969) \
                - 0.000005 * np.sin (606.9777 * T + 4.0212)  \
                - 0.000005 * np.sin (52.9691 * T + 0.4444)   \
                - 0.000002 * np.sin (21.3299 * T + 5.5431)   \
                - 0.000010 * T * np.sin (628.3076 * T + 4.2490)
        return ttmtdb


    def TTmTDB(self, tt_jd):
        """
        Time difference between TT and TDB.
        
        Interpolated using Chebyshev polynomials for an ephemeris file in
        TDB time that contains time scale transformation data
        (self.timescale == "TDB" and self.has_time). Otherwise it is calculated
        using TTmTDB_calc. Note that TDB on average runs at TT rate and this
        is a small correction of order 1ms. The accuracy of the correction is
        10 us.
        

        Parameters
        ----------
        tt_jd : float
                Julian time in the TT (terrestrial time) timescale.


        Returns
        -------
        float
                The difference TT-TDB for the TT time, given in seconds.
        """
        if self.timescale == "TDB":
            if self.has_time:
                return self.calc1(tt_jd, self.TTmTDB_ptr)[0][0]
        return Inpop.TTmTDB_calc(tt_jd)
    

    def TCGmTCB(self, tcg_jd):
        """
        Time difference between TCG and TCB.
        
        Only available for an ephemeris file in TCB time that contains time
        scale transformation data (self.timescale == "TCB" and self.has_time). 

        Parameters
        ----------
        tcg_jd : float
                 Julian time in the TCG (geocentric coordinate time) timescale.


        Returns
        -------
        float
                The difference TCG-TDB for the TCG time, given in seconds.
        """
        if not self.has_time:
            raise(LookupError("Ephemeris lacks time scale transformation."))
        if not self.timescale == "TCB":
            raise(LookupError("Ephemeris uses TDB time, not TCB."))
        return self.calc1(tcg_jd, self.TTmTDB_ptr)[0][0]


    @cnjit(signature_or_function='float64(float64)')
    def TCGmTT(tt_jd):
        """
        Time difference between TCG and TT.
        
        This difference is required if a TCB ephemeris is accessed through TT.
        TCG clocks run at a different rate than TT clocks.

        Parameters
        ----------
        tt_jd : np.double (or float)
                Julian time in the TT (terrestrial time) timescale.

        Returns
        -------
        float
                The difference TCG-TT for the TT time, given in seconds.

        """
        tai_jd = tt_jd - 32.184 / 86400
        tcgmtt = (Lg / (1 - Lg)) * (tai_jd - T0)
        return tcgmtt


    def close(self):
        """
        Close the INPOP file.

        Returns
        -------
        None.

        """
        if self.file:
            self.file.close()
        self.file=None


    def __del__(self):
        """
        Destructor, closes the INPOP file (if open).

        Returns
        -------
        None.

        """
        self.close()