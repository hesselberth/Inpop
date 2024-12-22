#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compact library for reading Inpop files.

Created on Fri Dec  4 20:10:07 2020

@author: Marcel Hesselberth

Version: 0.1
"""

from sys import byteorder
from os import path, SEEK_END
import struct
import numpy as np

from cnumba import cnjit, timer

bodycodes = {"mercury":0, "venus":1, "earth":2, "mars":3, "jupiter":4,
             "saturn":5, "uranus":6, "neptune":7, "pluto":8, "moon":9,
                          "sun":10, "ssb":11, "emb":12}


@cnjit(signature_or_function = 'UniTuple(float64[:], 2)(float64, int64)')
def chpoly(x, degree):
    """     
    Evaluate the Chebyshev polynomial and its derivatives at x.
    
    Best algorithm according to https://arxiv.org/abs/1312.5677v2

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


@cnjit(signature_or_function = 'f8[:, :](f8, i4, i4, i4, f8[:], f8, f8, i4, i4)')
def calcm(jd, offset, ncoeffs, ngranules, data, jd_beg, interval, nrecords, recordsize):
    """
    Calculate a 3 vector and its derivative from memory data.

    This is the Inpop decoding routine common to the calculations, whether
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
    if record < nrecords: record += 1 # not for the last. TODO: test
    raddr = record*recordsize
    jdl = data[raddr]
    jdh = data[raddr+1]
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
    def __init__(self, path, load=True):
        self.path = path
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
        Open the binary Inpop file.
        
        Read the header information, the constant values and initialize the
        lookup of Chebyshev polynomials. Some important variables are always
        present (AU, EMRAT, DENUM) and become member variables. According to
        specification 2.0 Inpop files can also contain asteroid information.
        Such files are not in the public domain, hence retrieving asteroid
        information is not (yet) implemented.

        Returns
        -------
        None.

        """
        self.file = open(self.path, 'rb')  # inpop files are binary
        
        # Decode the header record
        header_spec   = f"{self.byteorder}252s2400sdddidd36ii3ii3i"
        header_struct = struct.Struct(header_spec)
        bytestr       = self.file.read(header_struct.size)
        hb            = header_struct.unpack(bytestr)  # header block
        self.DENUM    = hb[44]  # must be 100 for inpop
        if self.DENUM != 100:
            self.file.seek(0)
            self.byteorder = self.opposite_byteorder
            header_spec    = f"{self.byteorder}252s2400sdddidd36ii3ii3i"
            header_struct  = struct.Struct(header_spec)
            bytestr        = self.file.read(header_struct.size)
            hb             = header_struct.unpack(bytestr)  # header block
            self.DENUM     = hb[44]
            if self.DENUM  != 100:
                raise(IOError("Can't determine Inpop file byteorder."))

        self.jd_struct  = struct.Struct(f"{self.byteorder}dd") # julian dates

        self.label      = []  # ephemeris label
        self.label.append(hb[0][:84].decode().strip())
        self.label.append(hb[0][84:168].decode().strip())
        self.label.append(hb[0][168:].decode().strip())

        const_names     = [hb[1][6*i:6*(i+1)] for i in range(400)]

        self.jd_beg     = hb[2]  # julian start date
        self.jd_end     = hb[3]  # julian end date
        self.interval   = hb[4]  # julian interval
        self.num_const  = hb[5]  # number of constants in the second record
        self.AU         = hb[6]  # Astronomical unit
        self.EMRAT      = hb[7]  # Mearth / Mmoon
        self.coeff_ptr  = [(hb[8+3*i:8+3*i+3]) for i in range(12)]
        self.DENUM      = hb[44]  # ephemeris ID
        self.librat_ptr = hb[45:48]  # libration pointer
        self.recordsize = hb[48]  # size of the record in bytes
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

        # conversion to a single unit base.
        self.unit_time = "s"
        self.unit_pos  = "au"
        self.unit_vel  = "au/day"

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
        
        self.earthfactor = -1 / (1 + self.EMRAT)
        self.moonfactor  = self.EMRAT / (1 + self.EMRAT)
        if self.mem:
            self.load()
            self.file.close()


    def load(self):
        from os import fstat
        self.file.seek(0, SEEK_END)
        size = self.file.tell()
        self.file.seek(0)
        if size % 8 != 0:
            raise(FormatError("INPOP File has wrong length."))
        data = np.frombuffer(self.file.read(size), dtype=np.double)
        data = data.newbyteorder(self.byteorder)
        self.data = np.copy(data)


    def info(self):
        s  = f"Inpop file             {self.path}\n"
        s += f"Label                  {self.label}\n"
        s += f"JDbeg, JDend, interval {self.jd_beg}, {self.jd_end}, {self.interval}\n"
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
        
        s += f"unit_time              {self.unit_time}\n"
        s += f"unit_pos               {self.unit_pos}\n"
        s += f"unit_vel               {self.unit_vel}\n"
        s += f"timescale              {self.timescale}"
        s += f"\ncoeff_ptr:\n{self.coeff_ptr}"
        return s


    def __str__(self):
        return self.info()


    def calc1(self, jd, coeff_ptr):
        """
        Calculate a state vector for a single body and its derivative.

        This is the Inpop decoding routine common to the calculations, whether
        6d (position-velocity), 3d (libration angles) or 1d (time).
        The file record is located and checked and subsequently the Inpop
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
        offset, ncoeffs, ngranules = coeff_ptr
        if self.mem:
            return calcm(jd, offset, ncoeffs, ngranules, \
                         self.data, self.jd_beg, self.interval, \
                         self.nrecords, self.recordsize)
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
        return np.array([pos, vel])


    def _PV(self, jd: float, body: int) -> np.ndarray:
        if not self.file:
            raise(IOError(f"Ephemeris file ({self.filename}) not open."))
        if body == 11:
            return np.zeros(6).reshape((2, 3))
        if body == 12:
            body = 2
        if body< 0 or body > 10:
            raise(LookupError("Body code must be between 0 and 12"))
        if jd < self.jd_beg or jd > self.jd_end:
            raise(ValueError("Julian date must be between %.1f and %.1f" \
                             % (self.jd_beg, self.jd_end)))
        pv = self.calc1(jd, self.coeff_ptr[body])
        pv[0] *= self.unit_pos_factor
        pv[1] *= self.unit_vel_factor
        return pv


    def PV(self, jd, t, c= 11):
        """
        Position and velocity of a body in the ICRF.
        The public Inpop ephemerides do not contain the velocities so they
        must be computed from the derivative of the Chebyshev polynomial.

        Parameters
        ----------

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

        jd : np.double (or higher precision)
        
        Julian date in ephemeris time. Inpop is distributed in TDB and TCB.
        timescales. At this lowest level the time must be expressed in the
        ephemeris timescale (see self.timescale).

        Returns
        -------
        2x3 matrix [P, V].
        None upon faulure (no ephemeris file found, time outside ephemeris
        body code invalid.

        P and V are numpy 3-vectors of tyoe np.double. P is the ICRF position
        in au. V is the velocity along the ICRF axes in au/day.

        """
        if t == c:
            return self._PV(jd, 11)
        if t == 2:
            target = self._PV(jd, 9) * self.earthfactor
            if c == 9:
                center = self._PV(jd, 9) * self.moonfactor
            else:
                target += self._PV(jd, 2)
                center = self._PV(jd, c)
        elif t == 9:
            target = self._PV(jd, 9) * self.moonfactor
            if c == 2:
                center = self._PV(jd, 9) * self.earthfactor
            else:
                target += self._PV(jd, 2)
                center = self._PV(jd, c)
        else:
            target = self._PV(jd, t)
            if c == 2:
                center = self._PV(jd, 9) * self.earthfactor \
                    + self._PV(jd, 2)
            elif c == 9:
                center = self._PV(jd, 9) * self.moonfactor \
                    + self._PV(jd, 2)
            else:
                center = self._PV(jd, c)
        return target - center


    def librations(self, jd):
        """
        Physical librations of the moon.

        Parameters
        ----------
        jd : float
             Date in ephemeris time (TDB or TCB, see self.timescale)

        Returns
        -------
        np.array(3, dype="float")
             The 3 physical libration angles in radians
        """
        if not self.file:
            raise(IOError(f"Ephemeris file ({self.filename}) not open."))
        if jd < self.jd_beg or jd > self.jd_end:
            raise(ValueError("Julian date must be between %.1f and %.1f" \
                             % (self.jd_beg, self.jd_end)))
        return self.calc1(jd, self.librat_ptr)[0]


    def _dt(self, jd):
        """
        Time difference (in seconds) between 2 time scales.
        
        Use the Chebyshev polynomial coefficients in the Inpop file to
        compute the time difference. This can be used to convert time scales
        to ephemeris time.

        Parameters
        ----------
        jd : float
             Date in ephemeris time (TDB or TCB, see self.timescale)

        Returns
        -------
        float
             A time difference in seconds.
        """
        if not self.file:
            raise(IOError(f"Ephemeris file ({self.filename}) not open."))
        if not self.has_time:
            raise(LookupError("Ephemeris lacks time scale transformation"))
        if jd < self.jd_beg or jd > self.jd_end:
            raise(ValueError("Julian date must be between %.1f and %.1f" \
                             % (self.jd_beg, self.jd_end)))
        pv = self.calc1(jd, self.TTmTDB_ptr)
        return pv[0][0]


    @cnjit(signature_or_function='float64(float64)')
    def TTmTDB_calc(tt_jd):  # truncated at <10 us presicion
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
        if self.timescale == "TDB":
            if self.has_time:
                return self._dt(tt_jd)
        return Inpop.TTmTDB_calc(tt_jd)
    

    def TCGmTCB(self, jd):
        if not self.timescale == "TCB":
            raise(LookupError("Ephemeris uses TDB time, not TCB"))            
        return self._dt(jd)


    def TCGmTT(self, tt_jd):
        tai_jd = tt_jd - 32.184 / 86400
        tcgmtt = (Lg / (1 - Lg)) * (tai_jd - T0)
        return tcgmtt

    @timer
    def test(self, filename=None):
        if filename == None:
            dirname = path.dirname(self.path)
            filename = path.basename(self.path)
            parts = filename.split("_")
            filename = "testpo."+parts[0].upper()+"_"+parts[1].upper()
            filename = path.join(dirname, filename)
        file = open(filename)
        lines = file.readlines()
        file.close()
        test = False
        largest = 0
        ecount = 0  # split properties TODO
        for line in lines:
            line=line.strip()
            if test:
                denum, date, jd, t, c, x, ref = line.split()
                denum = int(denum)
                jd = float(jd)
                t = int(t) - 1
                c = int(c) - 1
                x = int(x) - 1
                ref = float(ref)
                result = self.PV(jd, t, c).reshape(6)[x]
                ttmtdb = self.TTmTDB(jd)
                ttmtdb_calc = Inpop.TTmTDB_calc(jd)
                t_error = abs(ttmtdb - ttmtdb_calc)
                if t_error > 1e-5:
                    print(t_error)
                error = (result - ref)
                if error > largest:
                    largest = error
                if abs(error) > 1e-12:
                    print(t, c, x, result, ref, error)
                    ecount += 1
            if line.upper() == "EOT":
                test = True
        print(ecount, largest)


    def close(self):
        if self.file:
            self.file.close()
        self.file=None


if __name__ == "__main__":
    #inpop = Inpop("inpop21a_TDB_m100_p100_tt.dat")
    #inpop = Inpop("inpop10a_m100_p100_littleendian.dat")
    inpop = Inpop("inpop21a_TDB_m100_p100_tt.dat")
    #for c in inpop.constants:
    #    print(c, inpop.constants[c])
    print()
    print(str(inpop))
    print((inpop.PV(2415282.5, 9)-inpop.PV(2415282.5, 1)))
    inpop.test()
    print(inpop.librations(2460669) *180/np.pi)
    print(inpop.TTmTDB(2415282.5), Inpop.TTmTDB_calc(2415282.5))
    inpop.close()
    
